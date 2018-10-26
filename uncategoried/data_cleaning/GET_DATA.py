from pymatgen import MPRester                                                     
import urllib.request                                                             
import json
import csv
                                                                                  
if __name__ == "__main__":                                                        
    MAPI_KEY = "zQIrujyVEwEhTBRW"  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
                                                                                  
    # fetch list of a list of all available materials                             
    with urllib.request.urlopen('https://www.materialsproject.org/rest/v1/materials//mids') as myurl:
        data = json.loads(myurl.read().decode())                                  
        material_ids = data['response'] # 75,000'ish material IDs are returned 
                                                                                  
                                                                                  
    with MPRester(MAPI_KEY) as m: # object for connecting to MP Rest interface 
        criteria={'material_id': {'$in':material_ids[:]}} # to avoid straining the servers, this is only using the first 4 materials
        properties=['energy', 'pretty_formula']            # list a few quanteties of interest
        data = m.query(criteria, properties)                                      
        print(data[:100])
       
    
    keys=data[0].keys()
    with open('first_dataset.csv','w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)