import datetime

class STPoint:
    def __init__(self, lat, lng, time):
        self.lat = lat
        self.lng = lng
        self.time = time

    def __str__(self):
        return '({}, {}, {})'.format(self.time.strftime('%m/%d %H:%M:%S'), self.lat, self.lng)
