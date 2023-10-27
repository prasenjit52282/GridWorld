class Block:
    sizeX=50 #px
    sizeY=50 #px
    viewsize=10 #blocks in beam view statesize is (2*viewsize+1,2*viewsize+1)

    def __init__(self):
        pass

    @staticmethod
    def setBlockSize(sizeX,sizeY):
        Block.sizeX=sizeX
        Block.sizeY=sizeY

    @staticmethod
    def getBlockSize():
        return(Block.sizeX,Block.sizeY)

    @staticmethod
    def setViewSize(size):
        Block.viewsize=size

    @staticmethod
    def getViewSize():
        return(Block.viewsize,Block.viewsize)