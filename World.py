class World:
    def __init__(self, config):
        self.current_time = 0  # In seconds
        self.time_interval = config.time_interval  # Time interval in minutes
        self.trains = []
        self.lines = []


    def add_line(self, line):
        self.lines.append(line)

    def add_train(self, train):
        self.trains.append(train)

    def init_train_to_line(self):
        for train in self.trains:
            if train.departure_time == 0:
                self.put_train_to_line(train)

    def get_stations(self):
        """
        Get all stations in the world.
        
        :return: List of Station objects.
        """
        stations = []
        for line in self.lines:
            stations += line.get_all_stations()
        return stations
    
    def get_lines(self):
        """
        Get all lines in the world.
        
        :return: List of Line objects.
        """
        return self.lines
    
    def get_trains(self):
        """
        Get all trains in the world.
        
        :return: List of Train objects.
        """
        return self.trains
    
    # 动态添加，到达列车的启程时间时，火车出发
    def put_train_to_line(self, train):
        line = train.line
        line.add_train(train,0)

    # 动态删除，到达列车的终点站时，火车停止
    def remove_train_from_line(self, train):
        for line in self.lines:
            if train in line.trains:
                line.remove_train(train)
                return
    
    def get_train_num(self):
        """
        Get the total number of trains in the world.
        
        :return: Total number of trains as an integer.
        """
        return sum([len(line.trains) for line in self.lines])
    
    def judge_all_on_line(self):
        """
        Judge whether all trains are on line.
        
        :return: True if all trains are on line, False otherwise.
        """
        all_trains = self.get_trains()
        for train in all_trains:
            if train.on_line == False:
                return False
        return True
    def step(self):
        """
        Advance the simulation by one time step.
        """
        self.current_time += self.time_interval
        # 动态添加火车
        if self.judge_all_on_line() is False:
            for train in self.trains:
                if train.on_line == False and train.departure_time <= self.current_time:
                    self.put_train_to_line(train)
        collision = False
        for line in self.lines:
            line.update()
            line_collision = line.check_collisions()
            if line_collision:
                collision = True

        return collision
    
    def get_current_time(self):
        return self.current_time