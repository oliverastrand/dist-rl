import numpy as np

class Memory:

    def __init__(self, max_size):
        self.max_size = max_size
        self.experience_list = []
        self.itter = 0

    def save(self, experience):
        if len(self.experience_list) < self.max_size:
            self.experience_list.append(experience)
        else:
            self.experience_list[self.itter] = experience
            self.itter = (self.itter + 1) % self.max_size

    def get_random_sample(self, size):
        ret_list = []
        drawn_indexes = []
        ind = np.random.randint(len(self.experience_list))
        for i in range(size):
            while ind in drawn_indexes:
                ind = np.random.randint(len(self.experience_list))

            ret_list.append(self.experience_list[ind])
            drawn_indexes.append(ind)

        return ret_list
