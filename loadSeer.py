#SEER database
# SEER data should be loaded into the Data sub-directory of this project.
#
#  .\Data
#    \incidence
#       read.seer.research.nov14.sas       <- Data Dictionary
#       *.txt                              <- Data files in fixed width text format
#    \populations
#
# regex to read data dictionary
# \s+@\s+([0-9]+)\s+([A-Z0-9_]*)\s+[$a-z]+([0-9]+)\.\s+/\* (.+?(?= \*/))

import re
import time
import os
#import sqlite3
import glob
import mysql.connector


from os.path import join, abspath, dirname
whereami = abspath(dirname(__file__))  

class LoadSeerData:

    def __init__(self, reload = True, testMode = False, verbose = True):
        
        self.path = join(whereami, 'data')        

        # used to read in data dictionary, used to parse actual data files.
        self.SeerDataDictRegexPat = '\s+@\s+([0-9]+)\s+([A-Z0-9_]*)\s+[$a-z]+([0-9]+)\.\s+/\* (.+?(?=\*/))'
        self.dataDictFieldNames = ['Offest', 'ColName', 'Length']
        self.colOffset = []
        self.colName = []
        self.colLength = []

        self.testMode = testMode
        self.verbose = verbose
        self.reload = reload
        # open connection to the database
        self.init_database()


    def init_database(self):
        try:
            #initialize database
            self.db_conn = mysql.connector.connect(host='localhost',
                            port=3306,user='admin', passwd='910929',
                            database="test")            
            self.db_cur = self.db_conn.cursor()
            self.db_cur.execute("SET SESSION MAX_EXECUTION_TIME=10000000")
            
            if self.reload:
                if self.verbose:
                    print('DROP TABLE IF EXISTS seer\n')
                self.db_cur.execute("DROP TABLE IF EXISTS seer")            

            if self.verbose:
                print('Database initialized\n')
        except Exception as e:
            print('ERROR connecting to the database: ' + e.strerror)
            raise(e)



    def load_data_dictionary(self, fname = r'read.seer.research.nov17.sas'):
        if self.verbose:
            print('Start Load of Data Dictionary\n')

        # TODO look into a better way to read this file, don't like the if elif structure
        with open( join(self.path, fname) ) as fDataDict:
            for line in fDataDict:
                fields = re.match(self.SeerDataDictRegexPat, line, re.IGNORECASE)
                
                if fields:
                    for x in range(4):
                        if x == 0:
                            self.colOffset.append(int(fields.groups()[x])-1)  # change to 0 offset, Data Dict starts at 1
                        elif x == 1:
                            self.colName.append(fields.groups()[x])
                        elif x == 2:
                            self.colLength.append(int(fields.groups()[x]))
        
        if self.verbose:
            print('Data Dictionary loaded with %d columns \n' % len(self.colName))


    # supports specific file or wildcard filename to import all data in one call.
    # path specified is off of the path sent in the constructor so actual filename will be self.path + fname
    def load_data(self, fname = r'BREAST.TXT'):
        try:
            self.load_data_dictionary()
        except Exception as e:
            print('ERROR loading data dictionary: ' + e.strerror)
            raise(e)

        if not (len(self.colOffset) == len(self.colLength) == len(self.colName)) and len(self.colName) > 0:
            raise('Bad Data Dictionary Data')

        # create the table in the db
    #    if self.reload:
        self.create_table()

        timeStart = time.process_time()

        totRows = 0       
        
        for fileName in glob.glob( join(self.path, fname) ):
            totRows += self.load_one_file(fileName)
            

        if self.verbose:
            print('\nLoading Data completed.\n Rows Imported: %d records in %.2f (total) seconds.\n Loaded %.1f records per/sec.\n')


    def load_one_file(self, fname):
        if self.verbose:
            print('Start Loading Data: {}\n'.format(fname))

        # Need to get the name of the SEER text file so we can store it into the SOURCE field.
        fileSource = os.path.basename(fname)
        fileSource = os.path.splitext(fileSource)[0]
        
        # pre-build the sql statement outside of the loop so it is only called once
        #   get list of field names for INSERT statement
        fieldList = ','.join(map(str, self.colName))

        command = 'INSERT INTO seer(SOURCE,' + fieldList + ') values (' + '%s,' * len(self.colName) + '%s)'
        
        
        # create variables needed in loop
        testSize = 4000
        testIndex = 0
        runTests = 10
        testResults = []
        batchSize = 10               # INSERT 1000 at a time.
        rowValues = []               # hold one records values
        multipleRowValues = []       # hold batchSize lists of rowValues to commit to DB in one transaction
        totRows = 0                   

        # open SEER fixed width text file
        with open(fname, 'r') as fData:
            
            testT0 = time.clock()

            for line in fData:
                totRows += 1
                rowValues = []
                rowValues.append(fileSource)  # first field is the SEER data file name i.e. breast or respir

                # iterate through all of the fields in the text file and store to rowValues list
                for fldNum in range(len(self.colOffset)):
                    field = line[self.colOffset[fldNum]:self.colOffset[fldNum]+self.colLength[fldNum]]
                    rowValues.append(field)

                # store this one row list of values to the list of lists for batch insert                
                multipleRowValues.append(rowValues)

                # commit to DB in batchSize batches to speed performance
                if totRows % testSize == 0:
                    print(totRows)
                    self.db_cur.executemany(command, multipleRowValues)
                    self.db_conn.commit()
                    multipleRowValues=[]


                    testResults.append( [ testSize, 
                                          time.clock() - testT0, 
                                          (testSize / (time.clock() - testT0)) ] )
                   
                    runTests -= 1
                    if runTests <= 0:
                        runTests = 10
                        testIndex += 1
                   
                        if testIndex >= testSize:
                            testIndex = 0
                    
                    testT0 = time.clock()

                # if in testMode, exit loop after 100 records are stored
               # if totRows > 100 and self.testMode:
                  #  self.db_cur.executemany(command, multipleRowValues)
                  #  self.db_conn.commit()
                   # multipleRowValues = []
                   # break
            
            if len(multipleRowValues) > 0:
                self.db_cur.executemany(command, multipleRowValues)
                self.db_conn.commit() 
                multipleRowValues = []
                

        if self.verbose:
            print('\nLoading Data file completed. Rows Imported: {} records\n'.format(totRows))

        testResults.sort()

        for t in testResults:
            print(' %5d %3.2f %6.0f' % (t[0], t[1], t[2]))

        return totRows


    def create_table(self):
        # Create the table from the fields read from data dictionary and stored in self.colName
        # Make colName list comma delimited
        tableStrList = []
        for index in range(len(self.colName)):
            tableStrList.append( self.colName[index] + " CHAR(" + str(self.colLength[index]) +")" )
        delimList = ','.join( tableStrList ) 
        #print 'create table seer(SOURCE CHAR(20) NOT NULL,' + delimList + ')'        
        # create the table
        self.db_cur.execute('create table seer(SOURCE CHAR(20) NOT NULL,' + delimList + ')')
                            
    def __str__(self, **kwargs):
        pass




if __name__ == '__main__':

    timeStart = time.process_time();
    seer = LoadSeerData(reload = True, testMode = False, verbose = True)
    p = seer.load_data()  # load one file
    #p = seer.load_data(r'incidence\yr1973_2015.seer9\*.txt')   # load all files
    print('Module Elapsed Time: ', time.process_time() - timeStart)