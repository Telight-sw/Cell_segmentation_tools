from tqdm import tqdm
import numpy.typing as npt
import numpy as np

def Jaccard_binary(testImg: npt.NDArray, refImg: npt.NDArray) -> float:
    if refImg.shape != testImg.shape:
        return 0

    binRefImg = refImg > 0
    binTestImg = testImg > 0

    intersectionSize = (binRefImg * binTestImg).sum()
    unionSize = (binRefImg + binTestImg).sum()
    Jaccard = float(intersectionSize) / float(unionSize)

    #print(f"|intersection|={intersectionSize} / |union|={unionSize}")

    return Jaccard


def Jaccard(testImg: npt.NDArray, testLabel: int, refImg: npt.NDArray, refLabel: int) -> float:
    return Jaccard_binary(testImg == testLabel, refImg == refLabel)


def Jaccard(testImg: npt.NDArray, refImg: npt.NDArray) -> tuple[ dict[int,float], dict[int,float] ]:
    """
    Returns two maps (dictionaries) that tell quality values (Jaccards) of individual segments.
    The first returned dictionary maps for (not necessarily all) 'testImg' labels and their quality,
    the second returned dictionary maps for _all_ 'refImg' labels and their quality.

    Attention! The first map may not have an answer for every label present in the image, which
    is a signal that the sought after 'testImg' label matched to no 'refImg' label, the 'testImg' label
    represent a false positive segment for which the Jaccard score would be exactly 0.0.
    Consider using 'add_missing_evals()' to have the map completed with the missing 'testImg' labels.
    """
    refValues = set()
    for px in refImg.flat:
        refValues.add(px)

    Jaccard = dict()          #  refImg label -> quality
    Jaccard_on_test = dict()  # testImg label -> quality

    # for all pixel values in the reference image
    for refVal in tqdm(refValues):
        #print(f"processing ref value = {refVal}")
        if refVal == 0:
            continue

        # 'refVal' reference label indices and its size
        refLabelIdx = np.where(refImg == refVal)
        refLabelIdxSize = len(refLabelIdx[0])
        #print(f"  of size {refLabelIdxSize} pxs")

        # determine test image (non-zero) pixels (and their counts)
        # that "overlap" with the 'refVal' reference label
        intersectingTestValuesHist = {-1:0} # to make sure the values() will never be empty
        for px in testImg[ refLabelIdx ]:
            if px == 0:
                continue
            cnt = intersectingTestValuesHist.get(px)
            cnt = 1 if cnt is None else cnt+1
            intersectingTestValuesHist[px] = cnt

        # determine if there is a test label that overlaps
        # in more than half of the reference label size
        largestTestLabelSize = max(intersectingTestValuesHist.values())
        #print(f"  intersecting with {len(intersectingTestValuesHist)} test labels")
        #print(f"  largest intersection being {largestTestLabelSize} pxs")
        if 2*largestTestLabelSize <= refLabelIdxSize:
            Jaccard[refVal] = 0.0
            continue

        for px in intersectingTestValuesHist.keys():
            if intersectingTestValuesHist[px] == largestTestLabelSize:
                #print(f"  largest test label = {px}")
                testLabelSize = (testImg == px).sum()

                intersectionSize = largestTestLabelSize
                unionSize = testLabelSize + refLabelIdxSize - intersectionSize

                Jaccard[refVal] = float(intersectionSize) / float(unionSize)
                Jaccard_on_test[px] = Jaccard[refVal]

                #print(f"|intersection|={intersectionSize} / |union|={unionSize}")
                continue

    return Jaccard_on_test, Jaccard


def Jaccard_average(Jaccards: dict[int,float]) -> tuple[float,float]:
    """
    Returns the average Jaccard value, and "recall" aka "coverage" that is
    the TP/(TP+FN) where TP is where Jaccard value > 0 else it is FN.
    """
    sum = 0.0
    nonZeros = 0
    for J in Jaccards.values():
        sum += float(J)
        nonZeros += 1 if J > 0 else 0

    return sum / float(len(Jaccards)), float(nonZeros) / float(len(Jaccards.values()))


def project_eval_into(label_img: npt.NDArray, Jaccards: dict[int,float]) -> npt.NDArray:
    scale_img = np.zeros(label_img.shape, dtype='float32')

    for l in Jaccards.keys():
        quality_value = 0.1 if Jaccards[l] == 0 else Jaccards[l]
        scale_img[ label_img == l ] = quality_value

    return scale_img

def add_missing_evals(testImg: npt.NDArray, Jaccards: dict[int,float]) -> None:
    all_labels = set()
    for px in testImg.flat:
        all_labels.add(px)

    for px in all_labels:
        if Jaccards.get(px) is None and px > 0:
            Jaccards[px] = 0.0


class JaccardsHistogram:
    def __init__(self):
        self.bins_cnt = 40
        self.reset()

    def reset(self):
        self.hist = dict()
        for k in range(self.bins_cnt+1):
            self.hist[ float(k)/float(self.bins_cnt) ] = 0

        self.jacc_sum = 0.0
        self.jacc_cnt = 0   # count of all Jaccards
        self.jacc_zero = 0  # count of only zero-valued Jaccards

    def add(self, one_Jaccard_value: float) -> None:
        idx = int( min(max(one_Jaccard_value,0.0),1.0) * float(self.bins_cnt) )
        idx = float(idx)/float(self.bins_cnt)
        self.hist[idx] += 1

        self.jacc_sum += one_Jaccard_value
        self.jacc_cnt += 1
        self.jacc_zero += 0 if one_Jaccard_value > 0 else 1

    def add_dict(self, Jaccards: dict[int,float]) -> None:
        for j in Jaccards.values():
            self.add(j)

    def report_average_Jaccard(self) -> float:
        """
        Returns the average Jaccard, and relative coverage of all masks;
        the later defined as num_of_non-zero-Jaccard-values / num_of_all-Jaccard-values.
        Thus, returns the pair: avg Jaccard, avg coverage.
        """
        return self.jacc_sum / float(self.jacc_cnt) if self.jacc_cnt > 0 else 0.0, \
               float(self.jacc_cnt-self.jacc_zero) / float(self.jacc_cnt) if self.jacc_cnt > 0 else 0.0


    def create_hist(self, Jaccards: dict[int,float]) -> dict[float,float]:
        """
        Creates a histogram from the map of labels to their quality.
        The histogram is created with exactly the same bins layout
        as is used for the reset(), add() and add_dict() functions.
        This is merely a "Java-static-like" utility function.
        """
        h = dict()
        for k in range(self.bins_cnt+1):
            h[ float(k)/float(self.bins_cnt) ] = 0

        for j in Jaccards.values():
            idx = int( min(max(j,0.0),1.0) * float(self.bins_cnt) )
            idx = float(idx)/float(self.bins_cnt)
            h[idx] += 1

        return h

