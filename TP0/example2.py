import random

class StochasticClimber(Agent):    # Inherit from class Agent
    def climb(self, steps=1):
        """ 
            Run the stochastic climber for the specified number of steps

            :param steps: The number of steps to take
        """

        for step in range(0,steps):
            # Explore around the climber
            util_n = self.evaluate_utility(offset=(0.0,self.step_size))
            util_s = self.evaluate_utility(offset=(0.0,-self.step_size))
            util_e = self.evaluate_utility(offset=(self.step_size,0.0))
            util_w = self.evaluate_utility(offset=(-self.step_size,0.0))
            chosen_path = random.choices(['n', 's', 'e', 'w'],  weights=[util_n, util_s, util_e, util_w])[0]
            
            if chosen_path == 'n':
                self.move_up()
            elif chosen_path == 's':              
                self.move_down()
            elif chosen_path == 'e':            
                self.move_right()
            elif chosen_path == 'w':            
                self.move_left()

