from typing import Any, Dict, List

import pandas as pd
from pathlib import Path
import requests
import json

class NRELDataSet:
    folder_path: Path

    active_sites: pd.DataFrame
    raw_fields: Dict[str, List[str]]

    def __init__(self, url="https://midcdmz.nrel.gov/apps", path : Path=Path("./")):
        self.url = url
        self.folder_path : Path = path


        with open(self.get_site_list_file_path(), "r") as f:
            sites_df = pd.read_csv(f)

            self.active_sites = sites_df[sites_df["ACTIVE"] == 1] # Make sure we've only got active sites
        
        with open(self.folder_path.joinpath("data/site_field_list.json"), "r") as f:
            self.raw_fields = json.load(f)
    
    def get_site_list_file_path(self):
        return self.folder_path.joinpath("data/active_sites.csv")
        # return self.folder_path.joinpath("nrel_site_list.csv")
    
    def get_mappings_file_path(self):
        return self.folder_path.joinpath("data/mappings.json")


    def get_sites(self) -> List[str]:
        return self.active_sites["STATION_ID"].to_list()
    
    def get_site_attributes(self, site_id:str) -> List[str]:
        if site_id in self.raw_fields:
            return self.raw_fields[site_id]
        
        return []
    
    def load_mappings(self, default={}) -> Dict[str, Dict[str, Any]]:
        if not self.get_mappings_file_path().exists():
            print(f"Couldn't find mapping file at path {self.get_mappings_file_path()}")

            return default
        
        with open(self.get_mappings_file_path(), "r") as f:
            return json.load(f)

    def save_mappings(self, mappings: Dict[str, Dict[str, Any]]):
        with open(self.get_mappings_file_path(), 'w') as f:
            json.dump(mappings, f, indent=4)

    def get_global_attributes(self):
        return ["temp", "wind_speed", "wind_direction", "ght", "humidity"]


def request_station_data(site: str, begin: str, end: str, url="https://midcdmz.nrel.gov/apps/data_api.pl"):
    query_params = f'?site={site}&begin={begin}&end={end}'
    
    full_url = url + query_params
    
    r = requests.get(full_url)
    
    return r.status_code, r.text

def request_station_fields(site: str, url="https://midcdmz.nrel.gov/apps/field_api.pl"):
    query_params = f'?{site}'

    r = requests.get(url + query_params)

    return r.status_code, r.text

def request_station_list(url="https://midcdmz.nrel.gov/apps/data_api_doc.pl?_idtextlist_"):
    r = requests.get(url)

    return r.status_code, r.text

def request_fields(site: str, url="https://midcdmz.nrel.gov/apps/field_api.pl"):
    query_params = f'?{site}'

    full_url = url + query_params

    r = requests.get(full_url)

    return r.status_code, r.text