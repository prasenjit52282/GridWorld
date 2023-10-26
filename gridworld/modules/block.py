class Block:
    sizeX=50 #px
    sizeY=50 #px
    def __init__(self):
        pass

    @staticmethod
    def setBlockSize(sizeX,sizeY):
        Block.sizeX=sizeX
        Block.sizeY=sizeY

    @staticmethod
    def getBlockSize():
        return(Block.sizeX,Block.sizeY)