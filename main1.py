# This is Chng Eng Siong' implementation of DTW
# using python speech MFCC features

import matplotlib.pyplot as plt
from   python_speech_features import mfcc
from   python_speech_features import logfbank
import scipy.io.wavfile as wav
from   CDTWQuery import C_DTWQuery2Query



def genMFCC_Feature(wavefilename):
    (rate,sig) = wav.read(wavefilename)
    mfcc_feat = mfcc(sig,rate)
    (nr,nc) = mfcc_feat.shape
    mfcc_feat = mfcc_feat[:,1:nc]
    return mfcc_feat.transpose()


def genFBank_Feature(wavefilename):
    (rate,sig) = wav.read(wavefilename)
    fbank_feat = 10*logfbank(sig, rate)
    (nr,nc) = fbank_feat.shape
    return  fbank_feat.transpose()


# The main program starts here
# The following program is ONLY to compare queries of the same utterance
# NOT comparing against the corpus.

query1       = genFBank_Feature("./queryHello1.wav")
query2       = genFBank_Feature("./queryHello6.wav")


demoQuery2Query = 1
if (demoQuery2Query):
    myDTQQuery2Query  = C_DTWQuery2Query(query1,query2)

    DistMatrix              = myDTQQuery2Query.get_DistMatrix()
    dtwMatrix               = myDTQQuery2Query.get_dtwMatrix()
    dtwTraceForward         = myDTQQuery2Query.get_TraceForwardMatrix()
    actTraceBackMatrix      = myDTQQuery2Query.get_TraceBackMatrix()

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.imshow(dtwMatrix, interpolation='nearest')
    ax2.imshow(dtwTraceForward, interpolation='nearest')
    ax3.imshow(actTraceBackMatrix, interpolation='nearest')
    f.subplots_adjust(hspace=0)
    plt.show()


print "Ended"