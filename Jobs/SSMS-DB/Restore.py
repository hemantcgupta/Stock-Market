# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:52:40 2024

@author: Hemant
"""

from SqlDatabase import DBRESTORE

if __name__ == "__main__":
    dbList = ['mkanalyzer', 'mkdaymaster', 'mkintervalmaster', 'mkintervalanalyzer', 'mkprediction', 'mkgrowwinfo', 'mkgrowwdaymaster', 'mkgrowwintervalmaster', 'dms']
    for dbName in dbList:
        DBRESTORE(dbName, './Backup')    

