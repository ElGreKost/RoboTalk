class PID:
    def init(self, P, I, D, N_filter):
        self.P = P
        self.I = I
        self.D = D
        self.N = N_filter
        self.oldError = 0
        self.newError = 0
        self.P_factor = 0
        self.D_factor = 0
        self.I_factor = 0
        self.value = 0
        self.dt = 0.032

    def PID_calc(self, Target, Current):
        self.newError = Target - Current
        self.P_factor = self.P * self.newError
        self.D_factor = (self.D * self.N * (self.newError - self.oldError) + self.D_factor)
        self.D_factor = self.D_factor / (self.dt * self.N + 1)
        self.value = (self.P_factor + self.I_factor + self.D_factor)
        self.I_factor = self.I_factor + (self.I * self.dt * self.newError)
        self.oldError = self.newError
        return self.value


    def PID_get(self):
        P_I_D = [self.P, self.I, self.D]
        return P_I_D

    def PID_tune(self, P , I, D):
        self.P = P
        self.I = I
        self.D = D
        return