# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:44:08 2024

@author: Hemant
"""


from SqlDatabase import DBBACKUP

if __name__ == "__main__":
    dbList = ['mkanalyzer', 'mkdaymaster', 'mkintervalmaster', 'mkintervalanalyzer', 'mkprediction']    
    for dbName in dbList:
        DBBACKUP(dbName, './Backup')    
    
    



