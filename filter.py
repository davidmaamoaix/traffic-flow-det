class FilterPID:

    def __init__(self, p, d, initial_value=0):
        self.params = (p, d)
        self.current = initial_value

    def update(self, new_value):
        offset = new_value - self.current
        p_output = offset * self.params[0]
        d_output = offset * self.params[1]

        self.current += p_output
        self.current += d_output

        return self.current