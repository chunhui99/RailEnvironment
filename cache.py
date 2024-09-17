    def _get_render_data(self):
        """
        Get current data for rendering (lines, stations, trains, passengers).
        """
        lines_data = []
        for line in self.world.get_lines():
            stations_positions = [(station.station_id, station.waiting_passengers) for station in line.stations]
            trains_positions = [(train.position_on_line, train.current_passengers) for train in line.trains]
            lines_data.append({
                'line_name': line.name,
                'stations': stations_positions,
                'trains': trains_positions,
                'line_length': line.line_length
            })
        return lines_data

    def render(self, mode='human', time_steps=100, interval=1000):
        """
        Render the entire environment dynamically over a number of time steps.

        :param time_steps: Number of time steps to render.
        :param interval: Time interval (in milliseconds) between frames.
        """
        # Get data for rendering
        lines_data = self._get_render_data()

        fig, ax = plt.subplots()

        # Plot the initial setup
        def init():
            ax.clear()
            ax.set_xlim(-100, 2000)
            ax.set_ylim(-100, 2000)
            ax.set_title("Train Scheduling Visualization")

            # Draw stations as points and show passenger count
            for line in lines_data:
                x, y = self._get_line_position(line['line_name'])
                for station, passengers in line['stations']:
                    ax.plot(x, y, 'ro')  # Plot station points as red circles
                    ax.text(x, y, f'{station}\n{passengers}p', color='blue', fontsize=12)  # Display passenger count

        def update(frame):
            ax.clear()
            ax.set_xlim(-100, 2000)
            ax.set_ylim(-100, 2000)

            # Update train positions and check for collisions
            for line in lines_data:
                for idx, (train_position, passengers) in enumerate(line['trains']):
                    train_x, train_y = self._get_train_position(line['line_name'], train_position)
                    ax.plot(train_x, train_y, 'gs', markersize=10)  # Plot trains as green squares
                    ax.text(train_x, train_y, f'Train{idx + 1}\n{passengers}p', fontsize=12)  # Show passengers

                    # Check for collision
                    for other_line in lines_data:
                        if other_line != line:
                            for other_train_position, _ in other_line['trains']:
                                other_train_x, other_train_y = self._get_train_position(other_line['line_name'], other_train_position)
                                if self._check_collision(train_x, train_y, other_train_x, other_train_y):
                                    ax.text(train_x, train_y, 'Collision!', color='red', fontsize=16)
                                    print(f"Collision detected at {train_x}, {train_y} between Train{idx + 1} on {line['line_name']} and another train.")

        # Create an animation
        ani = animation.FuncAnimation(fig, update, frames=time_steps, init_func=init, interval=interval)
        plt.show()

    # def render(self, mode='human'):
    #     """
    #     Render the environment.

    #     :param mode: Rendering mode.
    #     """
    #     if mode == 'human':
    #         print(f"Current Time: {self.world.current_time} minutes")
    #         for line in self.world.network.lines:
    #             for train in line.trains:
    #                 print(f"Train {train.id} on {line.name}: Position {train.position_on_line}m, "
    #                       f"Speed {train.speed} m/s, Passengers {train.current_passengers}")