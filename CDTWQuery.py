# This is class DTWQuery2Query
# it supports the DTW of two queries that are THE SAME utterance text
# and hence force DTW to be end to end !!!

import numpy as np
from numpy import linalg as LA

class C_DTWQuery2Query:
    def __init__(self, aFeat1, aFeat2):
        self.aFeat1     = aFeat1  # the first feature array
        self.aFeat2     = aFeat2  # the first feature array
        (self.nFeature1, self.nFrame1) = self.aFeat1.shape
        (self.nFeature2, self.nFrame2) = self.aFeat2.shape
        self.distMatrix         = np.zeros((self.nFrame1, self.nFrame2))
        self.dtwMatrix          = np.zeros((self.nFrame1, self.nFrame2))
        self.dtwTraceForwardMatrix = np.zeros((self.nFrame1, self.nFrame2))
        self.dtwTracebackMatrix = np.zeros((self.nFrame1, self.nFrame2))

        self.__gen_DistMatrix()
        self.__gen_DTW_Query2Query()
        self.__gen_DTWTraceBack_Query2Query()

    # The distance matrix of 2 feature stream
    def __gen_DistMatrix(self):
        for i_Feat1 in range(self.nFrame1):
            xFeat1 = self.aFeat1[:,i_Feat1]
            for j_Feat2 in range(self.nFrame2):
                xFeat2 = self.aFeat2[:,j_Feat2]
                self.distMatrix[i_Feat1,j_Feat2] = LA.norm(xFeat1-xFeat2)


    # The optionType 'Query2Query' means that the 2 input features ARE exactly the same query
    # hence we want to initialise row 1 of DTW matrix to be large values, and starting point to be bottom left and ending point top right
    def __gen_DTW_Query2Query(self):
        maxVal                  = self.distMatrix.max()
        self.dtwMatrix[0, 0]    = self.distMatrix[0, 0]

        for i_Feat1 in range(1,self.nFrame1):
            self.dtwMatrix[i_Feat1,0] = maxVal*i_Feat1

        for j_Feat2 in range(1,self.nFrame2):
            self.dtwMatrix[0,j_Feat2] = maxVal*j_Feat2

        for j_Feat2 in range(1, self.nFrame2):
            for i_Feat1 in range(1,self.nFrame1):
                whichPreviousValues = [self.dtwMatrix[i_Feat1-1,j_Feat2],  self.dtwMatrix[i_Feat1,j_Feat2-1], self.dtwMatrix[i_Feat1-1,j_Feat2-1]]
                minval = min(whichPreviousValues )
                minIdx = np.argmin(whichPreviousValues)

                self.dtwMatrix[i_Feat1, j_Feat2] = self.distMatrix[i_Feat1, j_Feat2] + minval
                self.dtwTraceForwardMatrix[i_Feat1, j_Feat2] = minIdx


    def __gen_DTWTraceBack_Query2Query(self):
        i_Feat1 = self.nFrame1-1
        j_Feat2 = self.nFrame2-1

        self.dtwTracebackMatrix[i_Feat1, j_Feat2] = 1;
        print "Threshold found = "  + str(self.dtwMatrix[i_Feat1,  j_Feat2]) +" >>>"

        while (i_Feat1 >=0):
             whichIdx = self.dtwTraceForwardMatrix[i_Feat1,j_Feat2]
             if whichIdx == 0:
                 i_Feat1 = i_Feat1-1

             if whichIdx == 1:
                j_Feat2 = j_Feat2-1

             if whichIdx == 2:
                 i_Feat1 = i_Feat1 - 1
                 j_Feat2 = j_Feat2 - 1

             self.dtwTracebackMatrix[i_Feat1, j_Feat2] = 1;

    # The public functions to use are just getting the found matrixes
    def get_DistMatrix(self):
        return self.distMatrix

    def get_dtwMatrix(self):
        return self.dtwMatrix

    def get_TraceForwardMatrix(self):
        return self.dtwTraceForwardMatrix

    def get_TraceBackMatrix(self):
        return self.dtwTracebackMatrix
