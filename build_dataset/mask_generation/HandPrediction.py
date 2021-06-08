import multiprocessing
import HandPredictionModel
import fnmatch
import os
import joblib
from pure_sklearn.map import convert_estimator
import sys

recording_name = str(sys.argv[1])
print("The arguments are: " , str(sys.argv))
clf = joblib.load('model.pkl')
clf_pure_predict = convert_estimator(clf)


def main():
    imgs = fnmatch.filter(os.listdir(recording_name + '/color_frames/'), '*.png')
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)
    pool.map(HandPredictionModel.create_mask, imgs)

if __name__ == "__main__":
    main()
    print("done")

