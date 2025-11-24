# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:44:08 2024

@author: Hemant
"""


from SqlDatabase import DBBACKUP

if __name__ == "__main__":
    dbList = ['mkdaymaster', 'mkintervalmaster', 'mkanalyzer', 'mkintervalanalyzer', 'mkprediction', 'mkgrowwinfo', 'mkgrowwdaymaster', 'mkgrowwintervalmaster', 'dms']
    for dbName in dbList:
        try:
            DBBACKUP(dbName, './Backup')    
        except Exception as e:
            print(e)
            continue
    
