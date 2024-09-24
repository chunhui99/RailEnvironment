import numpy as np
import random

class DualLine:
    def __init__(self, line_forward, line_backward):
        """
        Initialize a dual line.

        :param line_forward: Line object for the forward direction.
        :param line_backward: Line object for the backward direction.
        """
        self.line_forward = line_forward
        self.line_backward = line_backward
        self.line_backward.set_dual_line(line_forward)
        self.line_forward.set_dual_line(line_backward)


class Line:
    def __init__(self, world, name, line_position_y, forward = True):
        """
        Initialize a train line.

        :param name: Name or identifier of the line.
        """
        self.name = name
        self.world = world
        self.start_station = None
        self.end_station = None
        self.stations = []
        self.station_distances = 0  # {(station_a, station_b): distance, ...}
        self.line_length = 0
        self.trains_info = {}  # {Train object: position on line}
        self.trains = []
        self.line_position_y = line_position_y
        self.forward = forward
        self.seed = None

    def set_seed(self, seed=None):
        self.seed = seed

    def get_world(self):
        return self.world
    
    def get_current_time(self):
        return self.world.get_current_time()
    
    def update_trains(self):
        """
        Update the state of all trains on the line.

        """
        for train in self.trains:
            if train.protect:
                train.protect = False
                continue
            train.update()
            if train.arrive_at_final:
                # print('Changing!!')
                self.remove_train(train)
                dual_line = self.get_dual_line()
                train.reset(dual_line)
                dual_line.add_train(train, 0)

    def update(self):
        """
        Update the state of the line.

        """
        # Update stations
        for station in self.stations:
            station.update()

        # Update the state of all trains on the line
        self.update_trains()
        # get the current position of all trains
        for train in self.trains:
            self.trains_info[train] = self.line_length - train.position_on_line

    def set_dual_line(self, dual_line):
        """
        Set the dual line parameters.

        :param dual_line: DualLine object.
        """
        self.dual_line = dual_line

    def get_dual_line(self):
        """
        Get the dual line object.

        :return: DualLine object.
        """
        return self.dual_line
    
    def set_line(self, start_station, end_station, stations, station_distances, begin_distance, end_distance):
        """
        Set the line parameters.
        :param start_station: Starting Station object.
        :param end_station: Ending Station object.
        :param stations: Ordered list of Station objects along the line.
        :param station_distances: Dictionary with keys as (station_a.station_name, station_b.station_name) tuples and values as distances between them.
        :param begin_distance: Distance from the start position to the first station.
        :param end_distance: Distance from the last station to the end position.    
        """

        self.start_station = start_station
        self.end_station = end_station
        self.stations = stations
        self.station_distances = station_distances
        self.line_length = sum([station_distances.get((stations[i].get_station_name(), stations[i + 1].get_station_name())) for i in range(len(stations) - 1)]) + begin_distance + end_distance
        self.end_distance = end_distance
        self.begin_distance = begin_distance

    def get_empty_distance(self):
        """
        Get the distance from the start position to the first station.

        :return: Distance as a float.
        """
        if self.forward:
            return self.begin_distance
        else:
            return self.end_distance
    
    
    def get_end_station(self):
        """
        Get the end station of the line.

        :return: Station object.
        """
        return self.end_station
    
    def add_train(self, train, line_position):
        """
        Add a train to the line.

        :param train: Train object to be added.
        """
        self.trains.append(train)
        self.trains_info[train] = line_position 
        train.line = self
        train.on_line = True
        train.position_on_line = line_position

    def remove_train(self, train):
        """
        Remove a train from the line, both from the list (self.trains) and the dictionary (self.trains_info).

        :param train: Train object to be removed.
        """
        # 从 self.trains 列表中移除列车
        if train in self.trains:
            self.trains.remove(train)
        
        # 从 self.trains_info 字典中移除列车
        if train in self.trains_info:
            del self.trains_info[train]

    
    def get_next_station(self, current_station):
        """
        Get the next station from the current station based on the direction.

        :param current_station: Current Station object.
        :return: Next Station object or None if at the end.
        """
        try:
            index = self.stations.index(current_station)
            if index < len(self.stations) - 1:
                return self.stations[index + 1]
            return None  # End of the line
        except ValueError:
            print('No Station Found!')
            raise ValueError
        
    def get_previous_station(self, current_station):
        """
        Get the previous station from the current station based on the direction.

        :param current_station: Current Station object.
        :return: Previous Station object or None if at the start.
        """
        try:
            index = self.stations.index(current_station)
            if index > 0:
                return self.stations[index - 1]
            return None  # Start of the line
        except ValueError:
            print('No Station Found!')
            raise ValueError

    def get_distance_between_stations(self, station_a, station_b):
        """
        Get the distance between two specified stations on the line.

        :param station_a: Station object.
        :param station_b: Station object.
        :return: Distance as a float or None if stations are not consecutive.
        """
        return self.station_distances.get((station_a.get_station_name(), station_b.get_station_name())) or self.station_distances.get((station_b.get_station_name(), station_a.get_station_name()))

    def get_all_stations(self):
        """
        Get the list of all stations on the line.

        :return: List of Station objects.
        """
        return self.stations

    def get_line_length(self):
        """
        Get the total length of the line.

        :return: Line length as a float.
        """
        return self.line_length

    def check_collisions(self):
        """
        Check for potential collisions between trains on the line.

        :return: Boolean indicating if any collision has occurred.
        """
        positions = sorted([(train.position_on_line, train) for train in self.trains], key=lambda x: x[0])
        # print('------------------------------------')
        # print(f"Positions: {positions}")
        # print('------------------------------------')
        for i in range(len(positions) - 1):
            current_train = positions[i][1]
            next_train = positions[i + 1][1]
            if next_train.position_on_line - current_train.position_on_line < next_train.length:
                print(f"Collision detected between Train {current_train.train_id} and Train {next_train.train_id} on Line {self.name}")
                return True
        return False

    def __repr__(self):
        return f"Line({self.name}, Stations: {[station.station_name for station in self.stations]})"
