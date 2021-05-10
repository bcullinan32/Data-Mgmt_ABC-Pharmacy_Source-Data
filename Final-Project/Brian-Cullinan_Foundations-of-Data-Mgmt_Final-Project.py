# Import Libraries
import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

# ## Prep -- Create DB from Source Files
# Create DB
path = 'C:/Users/TheCu/OneDrive/Documents/Grad-School-Docs/Foundations-of-DM/Data/ABCPharmacy/'
dbName = 'ABC_PharmacyDB.sqlite'


def createDB(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    createDB(path + dbName)

# Set DB Connection and cursor
db = sqlite3.connect(path + dbName)
cursor = db.cursor()

# Drop tables if they exist
cursor.execute("DROP table IF EXISTS phrmcy_master;")
cursor.execute("DROP table IF EXISTS prod_master;")
cursor.execute("DROP table IF EXISTS major_prod_cat;")
cursor.execute("DROP table IF EXISTS prod_cat;")
cursor.execute("DROP table IF EXISTS prod_sub_cat;")
cursor.execute("DROP table IF EXISTS prod_seg;")
cursor.execute("DROP table IF EXISTS pos_trans;")
cursor.execute("DROP table IF EXISTS sales;")
cursor.execute("DROP table IF EXISTS pos_trans_wTob;")

# Commit your changes in the database
db.commit()

# Create Tables
phrmcy_master = '''CREATE TABLE phrmcy_master (
          phrmcy_nbr FLOAT(30) NOT NULL,
          PHRMCY_NAM VARCHAR,
          st_cd CHAR(2),
          zip_3_cd INT(5),
          PRIMARY KEY (phrmcy_nbr, phrmcy_nam)
)'''
cursor.execute(phrmcy_master)

prod_master = '''CREATE TABLE prod_master (
          prod_nbr BIGINT(40) NOT NULL,
          prod_desc VARCHAR,
          segment_cd FLOAT(30),
          PRIMARY KEY (prod_nbr)
)'''
cursor.execute(prod_master)

major_prod_cat = '''CREATE TABLE major_prod_cat (
          major_cat_cd INT(4) NOT NULL,
          major_cat_desc VARCHAR,
          PRIMARY KEY (major_cat_cd)
)'''
cursor.execute(major_prod_cat)

prod_cat = '''CREATE TABLE prod_cat (
          cat_cd INT(5) NOT NULL,
          cat_desc VARCHAR NOT NULL,
          major_cat_cd INT(4),
          PRIMARY KEY (cat_cd)
          FOREIGN KEY (major_cat_cd) REFERENCES major_prod_cat(major_cat_cd)
)'''
cursor.execute(prod_cat)

prod_sub_cat = '''CREATE TABLE prod_sub_cat (
          sub_cat_cd INT(10) NOT NULL,
          sub_cat_desc VARCHAR,
          cat_cd INT(5),
          PRIMARY KEY (sub_cat_cd)
          FOREIGN KEY (cat_cd) REFERENCES prod_cat(cat_cd)
)'''
cursor.execute(prod_sub_cat)

prod_seg = '''CREATE TABLE prod_seg (
          seg_cd FLOAT(30) NOT NULL,
          seg_desc varchar,
          sub_cat_cd INT(10),
          PRIMARY KEY (seg_cd)
          FOREIGN KEY (sub_cat_cd) REFERENCES prod_sub_cat(sub_cat_cd)
          FOREIGN KEY (seg_cd) REFERENCES prod_master(segment_cd)
)'''
cursor.execute(prod_seg)

pos_trans = '''CREATE TABLE pos_trans (
          bskt_id BIGINT(40),
          phrmcy_nbr FLOAT(30),
          prod_nbr BIGINT(40),
          sls_dte_nbr INT(12),
          ext_sls_amt FLOAT(2),
          sls_qty INTEGER,
          FOREIGN KEY (phrmcy_nbr) REFERENCES phrmcy_master(phrmcy_nbr)
          FOREIGN KEY (prod_nbr) REFERENCES prod_master(prod_nbr)
)'''
cursor.execute(pos_trans)

pos_trans_wTob = '''CREATE TABLE pos_trans_wTob (
          sales_id INTEGER,
          bskt_id BIGINT(40),
          phrmcy_nbr FLOAT(30),
          prod_nbr BIGINT(40),
          sls_dte_nbr INT(12),
          ext_sls_amt FLOAT(2),
          sls_qty INTEGER,
          is_tob INT(1),
          FOREIGN KEY (phrmcy_nbr) REFERENCES phrmcy_master(phrmcy_nbr)
          FOREIGN KEY (prod_nbr) REFERENCES prod_master(prod_nbr)
)'''
cursor.execute(pos_trans_wTob)

# Commit your changes in the database
db.commit()

# read source csv files
path = 'C:/Users/TheCu/OneDrive/Documents/Grad-School-Docs/Foundations-of-DM/Data/ABCPharmacy/'

phrmcy_master = pd.read_csv(path + 'PHRMCY_MASTER.csv')
prod_master = pd.read_csv(path + 'PROD_MASTER.csv')
major_prod_cat = pd.read_csv(path + 'MAJOR_PROD_CAT.csv')
prod_cat = pd.read_csv(path + 'PROD_CAT.csv')
prod_sub_cat = pd.read_csv(path + 'PROD_SUB_CAT.csv')
prod_seg = pd.read_csv(path + 'PROD_SEG.csv')
pos_trans = pd.read_csv(path + 'POS_TRANS.csv')

# Populate Tables
phrmcy_master.to_sql('phrmcy_master', db, if_exists='append', index=False)
prod_master.to_sql('prod_master', db, if_exists='replace', index=False)
major_prod_cat.to_sql('major_prod_cat', db, if_exists='append', index=False)
prod_cat.to_sql('prod_cat', db, if_exists='append', index=False)
prod_sub_cat.to_sql('prod_sub_cat', db, if_exists='append', index=False)
prod_seg.to_sql('prod_seg', db, if_exists='append', index=False)
pos_trans.to_sql('pos_trans', db, if_exists='append', index=False)

# Update pos_trans_wTob
# prep pos_trans_wTob dataframe -- Create from orginal dataframe
pos_trans_wTob = pos_trans

# add 0 as default for is_tob
isTob = 0
pos_trans_wTob['is_tob'] = isTob

# Generate a Sales ID for each unique combination of BSKT_ID and SLS_DTE_NBR.  SALES_ID will not/ should not be unique within the date frame.
pos_trans_wTob['SALES_ID'] = (pos_trans_wTob.fillna({'BSKT_ID': '', 'SLS_DTE_NBR': ''})
                              .groupby(['BSKT_ID', 'SLS_DTE_NBR'], sort=False).ngroup() + 1)

# Augment bskt_ID
pos_trans_wTob['BSKT_ID'] = "a" + pos_trans_wTob['BSKT_ID']

# populate pos_trans_wTob table
pos_trans_wTob.to_sql('pos_trans_wTob', db, if_exists='append', index=False)

# UPDATE Is_Tab to 1 for products that are tobacco
updateIsTob1 = '''UPDATE pos_trans_wTob
SET is_tob = 1
WHERE prod_nbr in (SELECT DISTINCT pm.PROD_NBR 
FROM prod_cat pc
JOIN prod_sub_cat psc on psc.cat_cd = pc.cat_cd
JOIN prod_seg ps on ps.sub_cat_cd = psc.sub_cat_cd
JOIN prod_master pm on pm.SEGMENT_CD = ps.seg_cd 
WHERE pc.cat_cd = 7100
AND psc.sub_cat_cd <> 71007140);
'''
cursor.execute(updateIsTob1)
db.commit()

# Close connection
db.close()
print("Database created")

# ## Prep -- Get Data from SQLite DB and Prepare Dataframes

# Set DB Connection and cursor
db = sqlite3.connect(
    'C:/Users/TheCu/OneDrive/Documents/Grad-School-Docs/Foundations-of-DM/Data/ABCPharmacy/ABC_PharmacyDB.sqlite')

# Get data from pos_trans_wTOB for sales that are not tobacco
sales_notTob_qry = """SELECT pm.PHRMCY_NAM, ptwt.phrmcy_nbr, ptwt.prod_nbr, (ptwt.ext_sls_amt * ptwt.sls_qty) as lineTotal 
FROM pos_trans_wTob ptwt 
JOIN phrmcy_master pm ON pm.phrmcy_nbr = ptwt.phrmcy_nbr
WHERE ptwt.is_tob = 0;
"""
sales_notTob = pd.read_sql_query(sales_notTob_qry, db)

# Add sales total calculated field and group by pharmacy 
# sales_notTob["sls_sum"] = sales_notTob["ext_sls_amt"] * sales_notTob["sls_qty"]
sales_notTob = sales_notTob.groupby(by=["PHRMCY_NAM"]).sum()
sales_notTob = sales_notTob.rename(columns={"lineTotal": "notTobSales"})

# Get data from pos_trans_wTOB for sales that are tobacco
sales_isTob_qry = """SELECT pm.PHRMCY_NAM, ptwt.phrmcy_nbr, ptwt.prod_nbr, (ptwt.ext_sls_amt * ptwt.sls_qty) as lineTotal 
FROM pos_trans_wTob ptwt 
JOIN phrmcy_master pm ON pm.phrmcy_nbr = ptwt.phrmcy_nbr
WHERE ptwt.is_tob = 1;
"""
sales_isTob = pd.read_sql_query(sales_isTob_qry, db)

# Add sales total calculated field and group by pharmacy 
sales_isTob = sales_isTob.groupby(by=["PHRMCY_NAM"]).sum()
sales_isTob = sales_isTob.rename(columns={"lineTotal": "tobSales"})

# Get data for items bought with tobacco
bkstsWTob_qry = """SELECT phm.PHRMCY_NAM, phm.phrmcy_nbr, phm.st_cd, ptwt.sales_id, ptwt.bskt_id, ptwt.sls_dte_nbr, ptwt.prod_nbr,
(ptwt.sls_qty * ptwt.ext_sls_amt) AS lineTotal,
pm.PROD_DESC, psc.sub_cat_desc, mpc.major_cat_desc
FROM pos_trans_wTob ptwt
JOIN prod_master pm ON pm.PROD_NBR = ptwt.prod_nbr
JOIN phrmcy_master phm ON phm.phrmcy_nbr = ptwt.phrmcy_nbr
JOIN prod_seg ps ON ps.seg_cd = pm.SEGMENT_CD
JOIN prod_sub_cat psc on psc.sub_cat_cd = ps.sub_cat_cd
JOIN prod_cat pc on pc.cat_cd = psc.cat_cd
JOIN major_prod_cat mpc on mpc.major_cat_cd = pc.major_cat_cd
WHERE ptwt.bskt_id IN (SELECT DISTINCT ptwt.bskt_id
FROM  pos_trans_wTob ptwt
WHERE is_tob = 1)
AND is_tob = 0
ORDER by ptwt.bskt_id ASC, sls_dte_nbr;
"""
bkstsWTob = pd.read_sql_query(bkstsWTob_qry, db)
# Get pharmacy location data
phrmcyLoc_qry = """SELECT pm.PHRMCY_NAM, pm.st_cd, pm.zip_3_cd 
FROM phrmcy_master pm
GROUP BY pm.PHRMCY_NAM;
"""
phrmcyLoc = pd.read_sql_query(phrmcyLoc_qry, db)

# create assoc. items subset dataframe
assocItems = bkstsWTob
assocItems = assocItems[["PHRMCY_NAM", "phrmcy_nbr", "lineTotal"]].copy()
assocItems = assocItems.rename(columns={"lineTotal": "assocItemSales"})
assocItems = assocItems.groupby(by=["PHRMCY_NAM"]).sum()

# join data frames
sales_sum = pd.merge(sales_notTob, sales_isTob, left_index=True, right_index=True, how='outer')

# refactor - join assoc. items
sales_sum = pd.merge(sales_sum, assocItems, left_index=True, right_index=True, how='outer')

# reduce fields
sales_sum = sales_sum[["notTobSales", "tobSales", "assocItemSales"]].copy()

# replace NaN
sales_sum['tobSales'] = sales_sum['tobSales'].fillna(0)
sales_sum['assocItemSales'] = sales_sum['assocItemSales'].fillna(0)

# remove assoc. items from Not_tob totals
sales_sum['notTobSales'] = sales_sum['notTobSales'] - sales_sum['assocItemSales']

# add total revenue
sales_sum['totalRevenue'] = sales_sum['notTobSales'].fillna(0) + sales_sum['tobSales'] + sales_sum['assocItemSales']

# add percent fields
sales_sum["perNotTob"] = (sales_sum["notTobSales"] / sales_sum['totalRevenue'])
sales_sum["perTob"] = (sales_sum["tobSales"] / sales_sum['totalRevenue'])
sales_sum["perAssocItems"] = (sales_sum["assocItemSales"] / sales_sum['totalRevenue'])

print("dataframes created")

# ## Section 1 -- Tobacco and Cigarette Sales Impact to Overall Revenue and Profit

# Calculate High Level Overall Revenue Stats
allSalesTotal = round(sales_sum["totalRevenue"].sum(skipna=True))
notTob_salesTotal = round(sales_sum["notTobSales"].sum(skipna=True))
isTob_salesTotal = round(sales_sum["tobSales"].sum(skipna=True))
assocItem_salesTotal = round(sales_sum["assocItemSales"].sum(skipna=True))
percentTob = round((isTob_salesTotal / allSalesTotal) * 100, 1)
percentTobwAssoc = round(((isTob_salesTotal + assocItem_salesTotal) / allSalesTotal) * 100, 1)

# Add Revenue Variables to list
revList = pd.Series(
    ["Revenue", allSalesTotal, notTob_salesTotal, isTob_salesTotal, assocItem_salesTotal, percentTob, percentTobwAssoc])

# Calculate High Level Overall Profit Stats

# Set Margins - Based on sources cited in report
overallProfMarg = 0.218
tobProfMarg = 0.161

# Calculate Profit Stats
overallProfit = round(allSalesTotal * overallProfMarg)
notTobProfit = round(notTob_salesTotal * overallProfMarg)
tobProfit = round(isTob_salesTotal * tobProfMarg)
assocItemProfit = round(assocItem_salesTotal * overallProfMarg)
percentProfTob = round((tobProfit / overallProfit) * 100, 1)
percentProfTobwAssoc = round(((tobProfit + assocItemProfit) / overallProfit) * 100, 1)

# Add Profit Variables to list
profList = pd.Series(
    ["Profit", overallProfit, notTobProfit, tobProfit, assocItemProfit, percentProfTob, percentProfTobwAssoc])

# Merge Lists in Dataframe
revProfReport = pd.DataFrame([list(revList), list(profList)],
                             columns=["Type", "Overall Total", "Not Tobacco Total", "Tobacco Total",
                                      "Assoc. Item Total",
                                      "Percent of Tobacco", "Percent of Tobacco w/ Assoc. Item"])
# Print Report
display(revProfReport)
print()

# Calculate Profit Losses
lostProf = (tobProfit + assocItemProfit)
perOfProfLost = round((lostProf / overallProfit) * 100, 1)
newOverallProf = notTobProfit

# Print Report
print("Profit Loss Report:")
print(f"Projected profit reduction for six months: $", lostProf)
print(f"Percent of profit lost:", perOfProfLost)
print(f"New projected six month profit: $", newOverallProf)
print()

# Percent of total revenue pie chart
salesTotalList = ([notTob_salesTotal, isTob_salesTotal, assocItem_salesTotal])
labels = ["Not Tobacco", "Tobacco", "Items w/ Tobacco"]

plt.pie(salesTotalList, labels=labels,
        autopct='%1.1f%%', shadow=True, radius=1.25)
plt.title('Percentage of Tobacco Revenue \nIncluding Associated Items', fontsize=16, fontweight="bold", loc="right")
plt.show()

# Percent of total revenue pie chart
profitTotalList = ([overallProfit, tobProfit, assocItemProfit])
labels = ["Not Tobacco", "Tobacco", "Items w/ Tobacco"]

plt.pie(profitTotalList, labels=labels,
        autopct='%1.1f%%', shadow=True, radius=1.25)
plt.title('Percentage of Tobacco Profit \nIncluding Associated Items', fontsize=16, fontweight="bold", loc="right")
plt.show()

# ## Section 2 -- Branch Level Impacts
# Show # of Branches that Do/ Do Not Sell Tobacco
numBranch_total = len(sales_sum)
numBranch_noTob = len(sales_sum[sales_sum['tobSales'] == 0])
numBranch_wTob = len(sales_sum[sales_sum['tobSales'] > 0])

print("Branches with and without tobacco sales:")
print(f"The total number of ABC Pharmacy branches:", numBranch_total)
print(f"The total number of ABC Pharmacy branches that do not sell tobacco:", numBranch_noTob)
print(f"The total number of ABC Pharmacy branches that do sell tobacco:", numBranch_wTob)
print()

# Prep for bar chart display - subset to branches that sell tobacco & group by
bl_sales_sum = sales_sum[["perNotTob", "perTob", "perAssocItems"]].copy()

bl_sales_sum = bl_sales_sum[bl_sales_sum['perTob'] > 0]
bl_sales_sum.sort_values(by=['perTob'], inplace=True, ascending=True)

bl_sales_sum.plot(kind='barh', stacked='true',
                  align='edge', figsize=(16, 12))
plt.title('Percentage of Tobacco Revenue by Branch', fontsize=16, fontweight="bold")
plt.xlabel('Percentage', fontsize=14)
plt.ylabel('Branch', fontsize=14)
plt.legend(loc='lower right')
plt.show()

# Categorize Branches

# assign risk level based on conditions
atRiskBranch = sales_sum

# Classify Risk Category based on % of Rev
medPerLower = 0.782
lowPerLower = 0.95

riskPerConditions = [
    (atRiskBranch['perTob'] == 0),
    (atRiskBranch['perNotTob'] < medPerLower),
    (atRiskBranch['perNotTob'] < lowPerLower) & (atRiskBranch['perNotTob'] > medPerLower),
    (atRiskBranch['perNotTob'] > lowPerLower),
]
riskPerCat = ['0 - No Risk', '3 - High', '2 - Medium', '1 - Low']

atRiskBranch['riskLevel'] = np.select(riskPerConditions, riskPerCat)

# Repot Risk Counts
countHighRisk = len(atRiskBranch[atRiskBranch['riskLevel'] == '3 - High'])
countMedRisk = len(atRiskBranch[atRiskBranch['riskLevel'] == '2 - Medium'])
countHLowRisk = len(atRiskBranch[atRiskBranch['riskLevel'] == '1 - Low'])
countNoRisk = len(atRiskBranch[atRiskBranch['riskLevel'] == '0 - No Risk'])

print("Branch Risk Categorizations:")
print(f"Number of high risk branches as defined by percent of revenue over", round((1 - medPerLower) * 100, 1),
      "% is:", countHighRisk)
print(f"Number of medium risk branches as defined by percent of revenue over", round((1 - lowPerLower) * 100, 1),
      "% but lower than", round((1 - medPerLower) * 100, 1), "% is:", countMedRisk)
print(f"Number of low risk branches as defined by percent of revenue less than", round((1 - lowPerLower) * 100, 1),
      "is:", countHLowRisk)
print(f"Number of no risk branches that do not sell tobacco is: ", countNoRisk)
print()

# Prepare Risk Summary
atRiskBranchSummary = atRiskBranch
atRiskBranchSummary = atRiskBranch[['notTobSales', 'tobSales', 'assocItemSales', 'riskLevel']].copy()
atRiskBranchSummary['revenueSum'] = round(
    atRiskBranchSummary['notTobSales'] + atRiskBranchSummary['tobSales'] + atRiskBranchSummary['assocItemSales'])
atRiskBranchSummary = atRiskBranchSummary.groupby(by=["riskLevel"]).sum()

# Round Fields
atRiskBranchSummary['notTobSales'] = round(atRiskBranchSummary['notTobSales'])
atRiskBranchSummary['tobSales'] = round(atRiskBranchSummary['tobSales'])
atRiskBranchSummary['assocItemSales'] = round(atRiskBranchSummary['assocItemSales'])

# Add Calculated Fields
atRiskBranchSummary['revenuePercent'] = round((atRiskBranchSummary['revenueSum'] / allSalesTotal * 100), 1)
atRiskBranchSummary['notTabPer'] = round((atRiskBranchSummary['notTobSales'] / atRiskBranchSummary['revenueSum'] * 100),
                                         1)
atRiskBranchSummary['tobPer'] = round((atRiskBranchSummary['tobSales'] / atRiskBranchSummary['revenueSum'] * 100), 1)
atRiskBranchSummary['assocItemPer'] = round(
    (atRiskBranchSummary['assocItemSales'] / atRiskBranchSummary['revenueSum'] * 100), 1)

# Format Display
display(atRiskBranchSummary)
print()

# Prep for bar chart display - subset to branches that sell tobacco & group by
rl_sales_sum = atRiskBranch[['notTobSales', 'tobSales', 'assocItemSales', 'riskLevel']].copy()
rl_sales_sum = rl_sales_sum.groupby(by=["riskLevel"]).sum()
rl_sales_sum.sort_values(by=['riskLevel'], inplace=True, ascending=False)

# Display bar chart
rl_sales_sum.plot(kind='bar', stacked='true',
                  align='edge', figsize=(12, 6))
plt.title('Total Revenue by Risk Level', fontsize=18, fontweight="bold")
plt.xlabel('Branch Risk Level', fontsize=16)
plt.ylabel('Revenue in Millions', fontsize=16)
plt.yticks(np.arange(0, 10000000, 1000000))
plt.grid()
plt.legend(loc='lower right')
plt.show()

# ### Deep Dive into High Risk Branches
# Merge Dataframes
phrmcyLoc.set_index(["PHRMCY_NAM"], inplace=True)
highRiskBranch = atRiskBranch[atRiskBranch['riskLevel'] == '3 - High']
highRiskBranch = pd.merge(highRiskBranch, phrmcyLoc, left_index=True, right_index=True, how='left')

# Prepare Summary Report
highRiskBranch = highRiskBranch[
    ["notTobSales", "tobSales", "assocItemSales", "totalRevenue", "perNotTob", "perTob", "perAssocItems",
     "st_cd"]].copy()
highRiskBranch['perNotTob'] = round(highRiskBranch['perNotTob'] * 100, 1)
highRiskBranch['perTob'] = round(highRiskBranch['perTob'] * 100, 1)
highRiskBranch['perAssocItems'] = round(highRiskBranch['perAssocItems'] * 100, 1)
highRiskBranch.sort_values(by=['perNotTob'], inplace=True, ascending=False)

# Round Fields
highRiskBranch['notTobSales'] = round(highRiskBranch['notTobSales'])
highRiskBranch['tobSales'] = round(highRiskBranch['tobSales'])
highRiskBranch['assocItemSales'] = round(highRiskBranch['assocItemSales'])
highRiskBranch['totalRevenue'] = round(highRiskBranch['totalRevenue'])

#display dataframe
display(highRiskBranch)
print()

# Prep for bar chart display - subset to branches that sell tobacco & group by
highRiskSalesPlot = highRiskBranch[["perNotTob", "perTob", "perAssocItems"]].copy()

highRiskSalesPlot = highRiskSalesPlot[highRiskSalesPlot['perTob'] >= 0.218]
highRiskSalesPlot.sort_values(by=['perNotTob'], inplace=True, ascending=False)

highRiskSalesPlot.plot(kind='barh', stacked='true',
                       align='edge', figsize=(16, 12))
plt.title('Percentage of Tobacco Revenue by High Risk Branch', fontsize=16, fontweight="bold")
plt.xlabel('Percentage', fontsize=14)
plt.ylabel('Branch', fontsize=14)
plt.legend(loc='lower right')
plt.show()

# High Risk Branch Profits

# prepare dataframe 
highRiskBranchProfit = highRiskBranch
highRiskBranchProfit = highRiskBranchProfit[["notTobSales", "tobSales", "assocItemSales", "totalRevenue"]].copy()

# Calculate Current Profit
highRiskBranchProfit['ProfitFromTob'] = highRiskBranchProfit['tobSales'] * tobProfMarg
highRiskBranchProfit['ProfitFromAssocItems'] = highRiskBranchProfit['assocItemSales'] * overallProfMarg
highRiskBranchProfit['ProfitFromNotTob'] = highRiskBranchProfit['notTobSales'] * overallProfMarg
highRiskBranchProfit['CurrentBranchProfit'] = round(
    highRiskBranchProfit['ProfitFromTob'] + highRiskBranchProfit['ProfitFromAssocItems'] + highRiskBranchProfit[
        'ProfitFromNotTob'])

# Calculate New Profit
highRiskBranchProfit['NewProfit'] = round(highRiskBranchProfit['ProfitFromNotTob'])

# Format and display report
highRiskBranchProfit = highRiskBranchProfit[["CurrentBranchProfit", "NewProfit"]].copy()
highRiskBranchProfit.sort_values(by=['NewProfit'], inplace=True, ascending=True)
display(highRiskBranchProfit)
print()

# Calculate Profit for High Risk Branches
highRiskCurrentProfit = round(highRiskBranchProfit["CurrentBranchProfit"].sum())
highRiskCurrentPer = round((highRiskCurrentProfit / overallProfit) * 100, 1)

# Calculate Updated Profit for High Risk Braches
highRiskNewProfit = round(highRiskBranchProfit["NewProfit"].sum())
highRiskNewPer = round((highRiskNewProfit / overallProfit) * 100, 1)

# Calculate Diffs
diffProfit = round(highRiskCurrentProfit - highRiskNewProfit)
diffProfitPer = round((highRiskCurrentPer - highRiskNewPer), 1)

# Print Report
print("Profit impact from High Risk Branches:")
print(f"Sum of current profit for High Risk Branches is $", highRiskCurrentProfit, "which accounts for",
      highRiskCurrentPer, "% of overall profit")
print(f"Sum of projected new profit for High Risk Branches is $", highRiskNewProfit, "which accounts for",
      highRiskNewPer, "% of overall profit")
print(f"This represents a loss of $", diffProfit, "and", diffProfitPer, "% in profit")
print()

# ## Section 3 -- Associated Item Sales

# Items sold most frequently with tobacco

# prepare dataframe
itemsWTob = bkstsWTob
itemsWTob = itemsWTob[itemsWTob['lineTotal'] > 0]
itemsWTob = itemsWTob[["sub_cat_desc", "lineTotal"]].copy()

# Group and Sum by Subcategory
itemsWTob = itemsWTob.groupby(by=["sub_cat_desc"], as_index=False).sum()

# prepare and display
itemsWTob.sort_values(by=['lineTotal'], inplace=True, ascending=True)
itemsWTob['lineTotal'].describe()

# Create Boxplot of Subcategory Revenue
green_diamond = dict(markerfacecolor='g', marker='D')

boxplot = itemsWTob.boxplot(column='lineTotal', vert=False, autorange=False, figsize=(18, 2), whis=[5, 95],
                            patch_artist=True, flierprops=green_diamond)
plt.xlim([-500, 41000])
plt.title("Revenue by Product Subcategory")
plt.xlabel("Revenue in Dollars")
plt.annotate("95% of product subcategories are \n to the left of this upper whisker", xy=(0, 1.25))
plt.show

# Deep Dive into highest revenue subcategories

# Set threshold for highest level subcategories
quantileLevel = 0.95
limit = round(itemsWTob['lineTotal'].quantile(quantileLevel), 2)
popItemsWTob = itemsWTob[itemsWTob['lineTotal'] > limit]

# Calculate Revenue
sumMostFreqProd = round(popItemsWTob['lineTotal'].sum(), 2)
perOfAssocItems = round((sumMostFreqProd / assocItem_salesTotal) * 100, 1)

print(f"These products above the", (quantileLevel * 100), "percentile in revenue account for $", sumMostFreqProd,
      "in revenue.\n"
      "These", round((1 - quantileLevel) * 100, 3), "% of products account for", perOfAssocItems, "%  of the total$ ",
      assocItem_salesTotal,
      "\nin associated product sales.")

# Display summary report
display(popItemsWTob)
print()

# ## Clean-up
db.close()
