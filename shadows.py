import sys
import cv2
import numpy as np
import os
import pickle
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
from PyQt4 import QtCore, QtGui, uic


qtCreatorFile = "shadows.ui" # Enter file here.
fileIsSelected = False

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class MyApp(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.fname = ''
        self.setupUi(self)
        self.browse.clicked.connect(self.selectFile)
        self.suspicious.clicked.connect(self.svmtrain)
        self.detect.clicked.connect(self.featurextraction)

    def selectFile(self):
        global fileIsSelected
        self.fname=QtGui.QFileDialog.getOpenFileName(filter='*.avi')
        self.textBrowser.setText(self.fname)
        print self.fname
        fileIsSelected = True
        self.feature_status.setText("Selected.")

    def featurextraction(self):
        if(not fileIsSelected):
            self.feature_status.setText("Please select a video file")
            return
        global isFeatureExtractionDone
        pbar=1
        self.status.setText("Please wait...")
        self.progressBar1.setValue(pbar)
        cap = cv2.VideoCapture(str(self.fname))
        orb = cv2.ORB_create()
        try:
            if not os.path.exists('data'):
        	os.makedirs('data')
        except OSError:
            print ('Error: Creating directory of data')
        currentFrame = 0
        ret=True
        while(ret):
            self.progressBar1.setValue(pbar)
            ret, frame = cap.read()
            name = './data/frame' + str(currentFrame) + '.jpg'
            self.feature_status.setText('Creating...' + name)
            print ('Creating...' + name)
            if ret == True:
        	img = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

            k = cv2.waitKey(30) & 0xff
            if k == 27:
        	break
            cv2.imwrite(name,img)
            kp = orb.detect(img,None)
            kp, des = orb.compute(img, kp)
            if currentFrame!=0:
            	print(des)
            	with open("features.csv", 'wb') as f:
        		np.savetxt(f,des,delimiter=",")
            img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0),flags=0)
            cv2.imshow('frame',img2)
            currentFrame += 1
            pbar=pbar+0.18
        cap.release()
        cv2.destroyAllWindows()
        self.feature_status.setText("Completed!")
        self.progressBar1.setValue(100)
        isFeatureExtractionDone = True

    def svmtrain(self):
        if(not fileIsSelected):
            self.activity_status.setText("Error : Please Extract Features.")
            return
        df = pd.read_csv('train.csv')
    	numpy_array = df.as_matrix()
    	new=0.0
    	d = pd.read_csv('target.csv',header = 0)
    	narray = d.as_matrix()
    	clf = svm.SVC(kernel = 'linear', C = 1)
    	clf.fit(numpy_array, narray.ravel())
    	svm_pkl_filename = 'data.pkl'
    	svm_model_pkl = open(svm_pkl_filename, 'wb')
    	pickle.dump(clf, svm_model_pkl)
    	svm_model_pkl.close()
        svm_pkl_filename = 'data.pkl'
    	svm1_model_pkl = open(svm_pkl_filename, 'rb')
    	svm1_model = pickle.load(svm1_model_pkl)
    	print "Loaded svm model :: ", svm1_model
    	d1=pd.read_csv('features.csv',header=0)
    	narr=d1.as_matrix()
    	count0=0
    	count1=0
    	for row in narr:
    		a=svm1_model.predict(row.ravel())
    		if a==1:
    			count1=count1+1
    		elif a!=1:
    			count0=count0+1
        self.progressBar2.setValue(100)
        self.activity_status.setText("Test Completed")
    	if count1>count0:
            self.status.setText("Suspicious activity detected!")
    		# print("suspicious")
    	else:
            self.status.setText("No suspicious activity was detected")
    		# print("not suspicious")

if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
