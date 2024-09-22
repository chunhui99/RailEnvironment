import random
import numpy as np

class Train:
    def __init__(self, config, train_id, line, capacity, default_speed, max_speed, departure_time, length):
        """
        Initialize a train.

        :param train_id: ID of the train.
        :param line: The line the train is currently on.
        :param capacity: Maximum passenger capacity of the train.
        :param speed: Speed of the train (distance per time unit).
        """
        # 系统设置
        self.config = config
        self.time_interval = config.time_interval
        self.current_time = 0
        self.departure_time = departure_time
        self.moving = True # 是否在移动,初始是在移动的
        self.arrive_at_final = False # 是否到达终点站
        self.length = length
        self.train_id = train_id
        self.line = line  # Current line the train is on
        self.target_station = line.get_all_stations()[0]  # Start at the first station
        self.capacity = capacity
        self.current_passengers = 0
        self.default_speed = default_speed
        self.max_speed = max_speed
        self.speed = default_speed
        self.distance_to_next_station = line.get_empty_distance() # 距离下一站的距离, 初始是到始发站的距离
        self.action = None
        self.protect = False # 如果保护，比如刚刚换线，说明执行了动作，就不更新
        # 开始的时候不在line上，到了departure_time之后，就在line上了
        self.position_on_line = 0
        self.on_line = False
        self.get_position_x()
        self.position_y = line.line_position_y
        self.action_dict = {
            0: "accelerate",   # 加速
            1: "decelerate",   # 减速
            2: "stay",         # 保持速度
            3: "depart",       # 离站出发
            4: "stay"          # 不出发，继续等待
        }
        self.seed = self.config.seed

    def get_position_x(self):
        if self.line.forward is True:
            self.position_x = self.position_on_line
        else:
            self.position_x = self.line.line_length - self.position_on_line

    def reset(self, line):
        self.line = line
        self.position_y = line.line_position_y
        self.target_station = line.get_all_stations()[0]
        self.arrive_at_final = False
        self.speed = self.default_speed
        self.position_on_line = 0
        self.current_passengers = 0
        self.distance_to_next_station = line.get_empty_distance()
        self.action = None
        self.get_position_x()
        self.protect = True # 如果保护，比如刚刚换线，说明执行了动作，就不更新

    def stay(self):
        pass
    
    def set_current_time(self, current_time):
        self.current_time = current_time

    def depart_from_station(self, speed):
        """
        Depart from the current station and start moving towards the next station.
        """
        assert not self.arrive_at_final, "Train has arrived at the final station."
        self.moving = True
        self.speed = speed

    def accelerate(self):
        """
        Accelerate the train, increasing its speed by the acceleration rate.
        """
        if self.speed < self.max_speed:
            self.speed += self.config.speed_delta 
            self.speed = min(self.speed, self.max_speed)

    def decelerate(self):
        """
        Decelerate the train, decreasing its speed by the deceleration rate.
        """
        if self.speed > self.config.min_speed:
            self.speed -= self.config.speed_delta
            self.speed = max(self.speed, self.config.min_speed)

    def move(self):
        """
        Move the train towards the next station.
        """
        assert self.moving, "Train is not moving."
        
        # 移动的距离是速度乘以时间间隔，但不能超过到下一站的剩余距离
        move_distance = min(self.speed * self.time_interval, self.distance_to_next_station)
        
        if self.distance_to_next_station > 0:
            # 减少到下一站的剩余距离
            self.distance_to_next_station -= move_distance
            # 更新列车在线上的位置
            self.position_on_line += move_distance
        else:
            # 列车到站，调用列车到站处理函数
            self.arrive_at_final, self.passengers_to_unload = self.arrive_at_station()

    def set_seed(self, seed=None):
        self.seed = seed

    def get_position_on_line(self):
        return self.position_on_line
    
    def update(self):
        print('update self.current_passengers:', self.current_passengers)
        """
        Take an action based on the current state.
        """
        action = self.action
        # print('action:', action)
        actions_indices = np.argmax(action, axis=-1)  
        action = np.vectorize(self.action_dict.get)(actions_indices)
        # print('action:', action)
        if self.moving is False:
            # assert action == "depart" or action == 'stay', "Invalid action."
            if action != "depart" or action != "stay":
                # action 从 depart 和 stay 中 随机选一个
                action = random.choice(["depart", "stay"])
        elif self.moving is True:
            # assert action == "accelerate" or action == "decelerate" or action == "stay", "Invalid action."
            if action != "accelerate" or action != "decelerate" or action != "stay":
                # action 从 accelerate, decelerate 和 stay 中 随机选一个
                action = random.choice(["accelerate", "decelerate", "stay"])
        if action == "accelerate":
            self.accelerate()
            self.move()
        elif action == "decelerate":
            self.decelerate()
            self.move()
        elif action == "stay":
            self.stay()
            if self.moving:
                self.move()
        elif action == "depart":
            self.depart_from_station(self.default_speed)
            self.move()
        else:
            raise ValueError("Invalid action.")
        # 获得当前的位置
        self.get_position_x()
        
        
    def get_unload_passengers(self, mode = 'naive'):
        # print('self.arrive_at_final:', self.arrive_at_final)
        # 根据剩余station的数量，平均一下
        if mode == 'naive':
            return self.current_passengers // self.get_stations_behind()
        
    # 获得列车后面还有几站
    def get_stations_behind(self):
        # print('self.line.stations:', self.line.stations)
        # print('self.target_station:', self.target_station.station_name)
        # print('self.line.stations.index(self.target_station):', self.line.stations.index(self.target_station))
        return len(self.line.stations) - self.line.stations.index(self.target_station)
    
    def arrive_at_station(self):
        """
        Handle the event of arriving at a station, including passenger exchange.
        """
        self.moving = False
        # Update the current station
        current_station = self.target_station

        # Unload passengers at the station
        passengers_to_unload = self.get_unload_passengers()
        print('passengers_to_unload:', passengers_to_unload)
        self.current_passengers = self.current_passengers - passengers_to_unload

        # Load passengers from the station (if there are waiting passengers)
        available_capacity = self.capacity - self.current_passengers
        passengers_to_load = min(available_capacity, current_station.waiting_passengers)
        current_station.remove_passengers(passengers_to_load)
        self.current_passengers += passengers_to_load

        # Move to the next station based on the direction
        next_station_index = self.line.stations.index(current_station) + 1
        if next_station_index < 0 or next_station_index >= len(self.line.stations):
            self.arrive_at_final = True
            return self.arrive_at_final, passengers_to_unload
        else:
            self.target_station = self.line.stations[next_station_index]
            self.distance_to_next_station = self.line.get_distance_between_stations(current_station, self.target_station)
            return self.arrive_at_final, passengers_to_unload

    # 默认change到返程线路, 这个不放在这个类里面，放在World里面
    # def change_line(self):
    #     """
    #     Handle the event of reaching the terminal station and changing to a new line.
    #     """
    #     # Here, you may need a decision-making process to choose the next line when reaching a terminal station
    #     if self.current_station.is_terminal:
    #         # Logic to choose the next line (based on your environment setup)
    #         pass

    # 到下一站的距离
    def get_distance_to_next_station(self):
        return self.distance_to_next_station
    
    # 到终点的距离
    def get_distance_to_final(self):
        if self.on_line == False:
            return self.line.get_line_length()
        return self.line.get_line_length() - self.line.trains_info[self]
    
    # 获取目标站点
    def get_target_station(self):
        return self.target_station
    
    # 获取容量
    def get_capacity(self):
        return self.capacity
    
    # 获取当前乘客数
    def get_current_passengers(self):
        return self.current_passengers
    
    # 获取目标站点的乘客数
    def get_target_station_passengers(self):
        return self.target_station.waiting_passengers
    
    # 获取与前一辆车的车距，如果没有，设置为极大
    def get_distance_to_previous_train(self):
        line = self.line
        # 获得当前的车辆的index
        if self not in line.trains:
            # 返回线路长度
            return line.get_line_length()
        index = line.trains.index(self)

        # 如果是第一辆车
        if index == 0:
            return line.get_line_length()
        else:
            return line.trains[index-1].get_distance_to_final() - self.get_distance_to_final()
    
    # 获得前一辆车的容量，如果没有，设置为0
    def get_previous_train_capacity(self):
        line = self.line
        # 获得当前的车辆的index
        if self not in line.trains:
            return 0
        index = line.trains.index(self)
        # 如果是第一辆车
        if index == 0:
            return 0
        else:
            return line.trains[index-1].get_capacity()