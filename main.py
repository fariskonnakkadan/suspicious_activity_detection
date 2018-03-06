from __future__ import division
import sys
import cv2
import numpy as np
import os, fnmatch
from os import walk
import pickle
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QThread
import datetime
import itertools
import csv

now = datetime.datetime.now()

uifile_1 = 'login.ui'
form_1, base_1 = uic.loadUiType(uifile_1)

uifile_2 = 'main.ui'
form_2, base_2 = uic.loadUiType(uifile_2)

uifile_3 = 'train.ui'
form_3, base_3 = uic.loadUiType(uifile_3)

uifile_4 = 'log.ui'
form_4, base_4 = uic.loadUiType(uifile_4)

uifile_5 = 'batch.ui'
form_5, base_5 = uic.loadUiType(uifile_5)

fileIsSelected = False

class BatchProcess(base_5, form_5):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        uifile_5.__init__(self)
        self.setupUi(self)
        self.batchCloseButton.clicked.connect(self.close)
        self.selectFolderButton.clicked.connect(self.selectFolder)
        self.startBatch.clicked.connect(self.process)
        self.clearListButton.clicked.connect(self.listClear)
        self.batchLog.clicked.connect(self.showLog)
        self.log = Log()
        self.files = []
        self.directory = ""
        self.root = ""

    def selectFolder(self):
        self.directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        print dir

        for self.root, dirs, self.files in os.walk(self.directory):
            for filename in self.files:
                if filename.endswith(('.mp4', '.avi')):
                    pass
        n=len(self.files)
        # print self.files
        # print self.root
        for i in range(n):
            self.listWidget.addItem(self.files[i])

    def listClear(self):
        self.files = []
        self.listWidget.clear()

    def showLog(self):
        self.log.show()

    def process(self, files):
        print self.files
        print self.root+"/"+self.files[0]
        open('batch.txt', 'w').close()
        for i in range(len(self.files)):
            self.currentFile.setText(str(self.root+"/"+self.files[i]))
            cap = cv2.VideoCapture(str(self.root+"/"+self.files[i]))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orb = cv2.ORB_create()
            currentFrame = 0
            ret=True
            pbar = 0
            self.progressBar.setValue(pbar)
            while(ret):
                self.progressBar.setValue(pbar)
                ret, frame = cap.read()
                if ret == True:
                    img = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                	break
                    kp = orb.detect(img,None)
                    kp, des = orb.compute(img, kp)
                if currentFrame!=0:
                	with open("features.csv", 'wb') as f:
            		          np.savetxt(f,des,delimiter=",")
                img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0),flags=0)
                cv2.imshow('Video Playback',img2)
                currentFrame += 1
                pbar=pbar+(100/length)
            cap.release()
            cv2.destroyAllWindows()
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
            # self.activity_status.setText("Test Completed")
            self.progressBar.setValue(100)
            print "count1 = "+str(count1)
            print "count0 = "+str(count0)
            if count1>count0:
                f = open('log.txt','a')
                f.write(str(now)+" : Suspicious activity detected in video :"+str(self.root+"/"+self.files[i])+"\n")
                f.close()
                f = open('batch.txt','a')
                f.write(str("Suspicious activity detected in video        : "+self.files[i])+"\n")
                f.close()
            else:
                f = open('log.txt','a')
                f.write(str(now)+" : No Suspicious activity detected in video :"+str(self.root+"/"+self.files[i])+"\n")
                f.close()
                f = open('batch.txt','a')
                f.write(str("No Suspicious activity detected in video : "+self.files[i])+"\n")
                f.close()
        f = open('batch.txt','r')
        data = f.read()
        QtGui.QMessageBox.information(self, 'Suspicious activity', data)



class Log(base_4, form_4):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        uifile_4.__init__(self)
        self.setupUi(self)
        self.logCloseButton.clicked.connect(self.close)
        self.logUpdateButton.clicked.connect(self.update)
        self.update()

    def update(self):
        f = open('log.txt','r')
        message = f.read()
        f.close()
        self.logBrowser.setText(message)

class Login(base_1, form_1):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        uifile_1.__init__(self)
        self.setupUi(self)
        self.login.clicked.connect(self.handleLogin)
        self.exitLogin.clicked.connect(self.close)

    def handleLogin(self):
        if (self.username.text() == 'admin' and
            self.password.text() == 'admin'):
            self.accept()
        else:
            QtGui.QMessageBox.warning(self, 'Error', 'Incorrect username or password')

class Train(base_3, form_3):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        uifile_3.__init__(self)
        self.setupUi(self)
        self.trainSelectFile.clicked.connect(self.openFile)
        self.trainExtractButton.clicked.connect(self.extract)
        self.trainTrainButton.clicked.connect(self.svmtrain)
        self.trainExitButton.clicked.connect(self.close)

    def openFile(self):
        global fileIsSelected
        self.fname=QtGui.QFileDialog.getOpenFileName(filter='*.avi *.mp4')
        self.trainTextbox.setText(self.fname)
        fileIsSelected = True

    def extract(self):
        if(not fileIsSelected):
            self.trainProgressWindow.setText("Please select a video file")
            return
        global isFeatureExtractionDone
        pbar=0
        self.trainProgressWindow.setText("Please wait...")
        self.trainProgressBar.setValue(pbar)
        cap = cv2.VideoCapture(str(self.fname))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orb = cv2.ORB_create()
        currentFrame = 0
        ret=True
        while(ret):
            self.trainProgressBar.setValue(pbar)
            ret, frame = cap.read()
            if ret == True:
        	img = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
            k = cv2.waitKey(30) & 0xff
            if k == 27:
        	break
            kp = orb.detect(img,None)
            kp, des = orb.compute(img, kp)
            if currentFrame!=0:
                self.trainProgressWindow.setText(str(des))
            	with open("extracted.csv", 'wb') as f:
        		np.savetxt(f,des,delimiter=",")
            img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0),flags=0)
            currentFrame += 1
            pbar=pbar+(100/length)
        cap.release()
        cv2.destroyAllWindows()
        #open csv file and check rows, copy all data to train.csv add 0or1 to target.csv
        self.trainProgressWindow.setText("Completed!")
        self.trainProgressBar.setValue(100)
        isFeatureExtractionDone = True


    def svmtrain(self):
        self.trainProgressBar.setRange(0,0)
        f_features = 'extracted.csv'
        f_train = 'train.csv'
        target = 'target.csv'
        with open(f_features, "rb") as f_input, open(f_train, "ab") as f_output:
            csv_input = csv.reader(f_input)
            csv.writer(f_output).writerows(csv_input)
            f_input.close()
            f_output.close()
        print "Copy Completed"
        #to know the number of rows in features file
        input_file = open("extracted.csv","r+")
        reader_file = csv.reader(input_file)
        value = len(list(input_file))
        print "Number of rows = "
        print value
        if self.radioButtonSuspicious.isChecked():
        #creating 0's for appending with target
            list1 = []
            for i in range(value):
                list1.append(1)
            print "1's created"
        elif self.radioButtonNotSuspicious.isChecked():
            list1 = []
            for i in range(value):
                list1.append(0)
            print "0's created"
        #appending list containing 0's on target
        with open("target.csv", "ab") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerows(zip(list1))
        print "append to target.csv"
        print "completed"
        input_file.close()
        fp.close()
        self.thread1 = Thread()
        self.thread1.start()



class MyApp(base_2, form_2, Log):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        uifile_2.__init__(self)
        self.fname = ''
        self.setupUi(self)
        self.browse.clicked.connect(self.selectFile)
        self.extract.clicked.connect(self.test)
        self.trainsvm.clicked.connect(self.svmTrainWindow)
        self.dialog = Train()
        self.logButton.clicked.connect(self.showLog)
        self.log = Log()
        self.exit.clicked.connect(self.close)
        self.batchButton.clicked.connect(self.batchWindow)
        self.batch = BatchProcess()

    def svmTrainWindow(self):
        self.dialog.show()

    def showLog(self):
        self.log.show()

    def batchWindow(self):
        self.batch.show()




    def selectFile(self):
        global fileIsSelected
        self.fname=QtGui.QFileDialog.getOpenFileName(filter='*.avi *.mp4')
        self.textBrowser.setText(self.fname)
        print self.fname
        fileIsSelected = True
        self.progressbox.setText("Selected.")

    def test(self):
        if(not fileIsSelected):
            self.progressbox.setText("Please select a video file")
            return
        global isFeatureExtractionDone
        pbar=0
        self.status.setText("Please wait...")
        self.progressBar1.setValue(pbar)
        self.progressBar2.setValue(0)
        cap = cv2.VideoCapture(str(self.fname))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orb = cv2.ORB_create()
        currentFrame = 0
        ret=True
        while(ret):
            self.progressBar1.setValue(pbar)
            ret, frame = cap.read()
            if ret == True:
        	img = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )
            k = cv2.waitKey(30) & 0xff
            if k == 27:
        	break
            kp = orb.detect(img,None)
            kp, des = orb.compute(img, kp)
            if currentFrame!=0:
                self.progressbox.setText(str(des))
            	with open("features.csv", 'wb') as f:
        		np.savetxt(f,des,delimiter=",")
            img2 = cv2.drawKeypoints(img,kp,img,color=(0,255,0),flags=0)
            cv2.imshow('Video Playback',img2)
            currentFrame += 1
            pbar=pbar+(100/length)
        cap.release()
        cv2.destroyAllWindows()
        self.progressbox.setText("Completed!")
        self.progressBar1.setValue(100)
        isFeatureExtractionDone = True

        svm_pkl_filename = 'data.pkl'
    	svm1_model_pkl = open(svm_pkl_filename, 'rb')
    	svm1_model = pickle.load(svm1_model_pkl)
    	print "Loaded svm model :: ", svm1_model
        self.progressbox.setText(str(svm1_model))
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
        # self.activity_status.setText("Test Completed")
        print "count1 = "+str(count1)
        print "count0 = "+str(count0)
    	if count1>count0:
            self.progressBar2.setValue(100)
            self.status.setText("Suspicious activity detected")
            QtGui.QMessageBox.warning(self, 'Warning', 'Suspicious Activity Detected')
            f = open('log.txt','a')
            f.write(str(now)+" : Suspicious activity detected in video :"+self.fname+"\n")
            f.close()
    	else:
            self.progressBar2.setValue(100)
            self.status.setText("No suspicious activity was detected")
            QtGui.QMessageBox.information(self, 'Done', 'No Suspicious Activity Detected')
            f = open('log.txt','a')
            f.write(str(now)+" : No Suspicious activity detected in video :"+self.fname+"\n")
            f.close()

class Thread(QThread):
    def __init__(self):
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        print "Thread started"
        df = pd.read_csv('train.csv')
    	numpy_array = df.as_matrix()
    	new=0.0
    	d = pd.read_csv('target.csv',header = 0)
    	narray = d.as_matrix()
        print "matrix completed"
        clf = svm.SVC(kernel = 'linear', C = 1)
    	clf.fit(numpy_array, narray.ravel())
    	svm_pkl_filename = 'data.pkl'
    	svm_model_pkl = open(svm_pkl_filename, 'wb')
    	pickle.dump(clf, svm_model_pkl)
    	svm_model_pkl.close()
        print "Training Completed"

if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    login = Login()


    if login.exec_() == QtGui.QDialog.Accepted:
        window = MyApp()
        window.show()
        sys.exit(app.exec_())
