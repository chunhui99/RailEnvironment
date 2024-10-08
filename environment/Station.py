import random
import numpy as np

# 可能可以抽象出一个该站点的抽象类，包含去向，来向，以及其他线该站点的情况

class Station:
    def __init__(self, config, station_id, station_name, line, max_capacity, passenger_arrival_time_bin, station_position_x, station_position_y, is_transfer = False):
        """
        Initialize a station.

        :param station_id: station id of the station.
        :param station_name: name of the station.
        :param line_info: dictionary where key is the line and value is the position index in that line.
        :param max_capacity: maximum passenger capacity of the station.
        :param is_transfer: boolean indicating if the station is a transfer station.
        """
        self.config = config
        self.time_interval = config.time_interval
        self.station_id = station_id
        self.station_name = station_name
        self.line = line
        self.station_position_x = station_position_x
        self.station_position_y = station_position_y
        self.passenger_arrival_time_bin = passenger_arrival_time_bin

        # 是否是终点站
        if self == self.line.get_end_station():
            self.is_terminal = True
        else:
            self.is_terminal = False
        
        # 是否是换乘站
        self.is_transfer = is_transfer

        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.passenger_groups = []  # List of tuples [(number of passengers, wait time)]
        self.waiting_passengers = 0
        self.current_time = 0
        self.total_wait_time = 0
        self.is_full = False

    def get_world(self):
        return self.line.world
    
    def get_current_time(self):
        return self.get_world().get_current_time()
    
    def update_wait_times(self):
        """
        Update the waiting time for each passenger group. Increase the wait time for each group by one time step.
        """
        for i in range(len(self.passenger_groups)):
            passengers, wait_time = self.passenger_groups[i]
            self.passenger_groups[i] = (passengers, wait_time + self.time_interval)  # Increment each group's wait time

    def add_passengers(self, num_passengers):
        """
        Add passengers to the station.
        If the station reaches its capacity, some passengers will not be able to enter.
        
        :param num_passengers: number of new passengers arriving at the station.
        """

        if self.current_capacity + num_passengers > self.max_capacity:
            accepted_passengers = self.max_capacity - self.current_capacity
            self.current_capacity = self.max_capacity
            self.is_full = True
        else:
            accepted_passengers = num_passengers
            self.current_capacity += num_passengers
            self.is_full = self.current_capacity >= self.max_capacity

        # Add the new group of passengers with a wait time of 0
        if accepted_passengers > 0:
            self.passenger_groups.append((accepted_passengers, 0))

        self.waiting_passengers += accepted_passengers

    def __deepcopy__(self, memo):
        import copy
        if id(self) in memo:
            return memo[id(self)]
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def remove_passengers(self, num_passengers):
        """
        Remove passengers from the station when they board a train, starting from those who have waited the longest.
        
        :param num_passengers: number of passengers boarding the train.
        """
        removed_passengers = 0
        total_removed_wait_time = 0
        # Sort the passenger groups by their waiting time (longest wait time first)
        self.passenger_groups.sort(key=lambda x: x[1], reverse=True)

        # Traverse the passenger groups and remove passengers starting with the group that waited the longest
        for i in range(len(self.passenger_groups)):
            passengers, wait_time = self.passenger_groups[i]
            if num_passengers <= 0:
                break
            if passengers <= num_passengers:
                removed_passengers += passengers
                total_removed_wait_time += passengers * wait_time
                num_passengers -= passengers
                self.passenger_groups[i] = (0, wait_time)  # Mark this group as empty
            else:
                # Remove part of the group
                removed_passengers += num_passengers
                total_removed_wait_time += num_passengers * wait_time
                self.passenger_groups[i] = (passengers - num_passengers, wait_time)
                break

        # Remove groups with 0 passengers
        self.passenger_groups = [(p, w) for p, w in self.passenger_groups if p > 0]

        # Update waiting passengers and capacity
        self.waiting_passengers -= removed_passengers
        self.current_capacity -= removed_passengers
        self.is_full = self.current_capacity >= self.max_capacity

        return total_removed_wait_time
    
    def get_total_wait_time(self):
        """
        Calculate and return the total wait time for all passengers at the station.
        """
        total_wait_time = 0
        for passengers, wait_time in self.passenger_groups:
            total_wait_time += passengers * wait_time
        return total_wait_time

    def get_waiting_passengers(self):
        """
        Get the number of passengers waiting at the station.
        
        :return: Number of passengers as an integer.
        """
        return self.waiting_passengers
    def update(self):
        """
        Update the station at each time step.
        """
        # print('self.waiting_passengers:', self.waiting_passengers)
        if not self.is_terminal:
            passengers = self.generate_passengers()
            # print('passengers:', passengers)
            self.update_wait_times()
            self.add_passengers(passengers)
        # print('self.waiting_passengers:', self.waiting_passengers)

    def generate_passengers(self):
        """
        Generate a random number of passengers arriving at the station.
        """
        if self.config.use_real_data == 0:
            self.current_time = self.get_current_time()
            single_minute = self.current_time % 60
            miniute_bin = single_minute // 6
            avg_passengers = self.passenger_arrival_time_bin[miniute_bin]
            # Set a random fluctuation, e.g., +/- 20% of the average passengers
            fluctuation_ratio = self.local_random.uniform(0.8, 1.2)  # Fluctuate by 20% up or down
            passengers = int(avg_passengers * fluctuation_ratio)
        else:
            self.current_time = self.get_current_time()
            if self == self.line.get_end_station():
                passengers = 0
            else:
                avg_passengers = self.config.real_data_dict[self.current_time + 1]  
                fluctuation_ratio = self.local_random.uniform(0.8, 1.2)  # Fluctuate by 20% up or down
                passengers = int(avg_passengers * fluctuation_ratio)

        return passengers
    
    def get_station_name(self):
        """
        Get the name of the station.
        
        :return: Name of the station as a string.
        """
        return self.station_name
    
    def set_seed(self,seed):
        
        self.seed = seed
        self.local_random = random.Random(seed)

        