import pandas as pd


first_img_set = pd.read_csv('/home/dhiraj/Desktop/via_region_data-0000-1644.csv')

knife_count= (first_img_set["region_attributes"].str.count("knife")).sum()
scissors_count= (first_img_set["region_attributes"].str.count("scissors")).sum()
print("Total no. of scissors in first_img_set: ",scissors_count)
print("Total no. of knife in first_img_set : ",knife_count)



