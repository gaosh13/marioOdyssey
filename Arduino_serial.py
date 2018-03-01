import serial

class Serial:
    def __init__(self, BAUD=57600, PORT="COM4"):
        self.ser = serial.Serial()
        self.ser.baudrate = BAUD
        self.ser.port = PORT
        self.ser.open()
        self.operation = ""
    
    def qsend(self, X, Y):
        if X != "" or Y != "":
            self.ser.write(bytes(X + Y, 'ascii'))
            # print(X + Y)

    def send(self):
        self.ser.write(bytes(self.operation, 'ascii'))

    def operate(self, X, Y):
        self.operation = X + " " + Y
    
    def close(self):
        self.ser.close()

class Serial_null:
    def __init__(self, BAUD=57600, PORT="COM4"):
        print("Serial test")
    
    def qsend(self, X, Y):
        pass
    
    def send(self):
        pass
    
    def operate(self, X, Y):
        pass
    
    def close(self):
        pass

class Serial_physical(Serial):
    def __init__(self, BAUD=57600, PORT="COM5"):
        super().__init__(BAUD, PORT)

    def qsend(self, X, Y):
        res = ""
        if X == "" and Y == "":
            res = "s"
        if X == "w":
            if Y == "":
                res = "w"
            elif Y == "a":
                res = "q"
            else:
                res = "e"
        elif X == "s":
            if Y == "":
                res = "x"
            elif Y == "a":
                res = "z"
            else:
                res = "c"
        else:
            res = Y
        
        self.ser.write(bytes(res, 'ascii'))