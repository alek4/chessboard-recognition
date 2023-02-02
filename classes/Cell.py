class Cell(object):
    def __init__(self, tl, tr, br, bl, coords, color):
        self.tl = (int(tl[0]), int(tl[1]))
        self.tr = (int(tr[0]), int(tr[1]))
        self.br = (int(br[0]), int(br[1]))
        self.bl = (int(bl[0]), int(bl[1]))
        self.color = color
        self.isEmpty = False
        self.wtl = None
        self.wtr = None
        self.wbr = None
        self.wbl = None

        self.center = self.calculateCentroid()

        self.coords = coords

        self.img = None

        self.score = None

    def calculateCentroid(self):
        cx = (self.tl[0] + self.br[0]) // 2
        cy = (self.tl[1] + self.br[1]) // 2

        return (cx, cy)

    def calculateWCentroid(self):
        cx = (self.wtl[0] + self.wbr[0]) // 2
        cy = (self.wtl[1] + self.wbr[1]) // 2

        return (cx, cy)
