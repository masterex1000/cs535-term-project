#!/bin/python3

# This is a small interactive terminal program that helps build the mappings between a data-set's attributes
# And our global attributes.

import os
import argparse
from typing import Dict, List
from collections import OrderedDict

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.database import Database
from pymongo.collection import Collection

from iterfzf import iterfzf
from tabulate import tabulate

from scsc_wrangle_utils import get_collection_attribute_types, get_collection_keys, get_dataset_metadata, get_global_attributes

mongo_url = 'mongodb://lattice-150.cs.colostate.edu:27018'

username = os.getenv("ROOT_MONGO_USER")
password = os.getenv("ROOT_MONGO_PASS")

mongo_client = MongoClient(mongo_url, username=username, password=password)

class AttributeMapping:
    def __init__(self, mapping=None, oldMapping = None, isOfficalMapping = True):
        self.mapping: dict = mapping
        self.oldMapping: dict = oldMapping
        self.isOffical: bool = isOfficalMapping # Decides if the mapping is an offical value or not

        if self.mapping == None:
            self.mapping = oldMapping

    def get_mapped_name(self) -> str:
        try:
            return self.mapping["localKey"]
        except:
            return None

    def set_mapped_name(self, newName: str):
        if self.mapping == None:
            self.mapping = {}

        self.mapping["localKey"] = newName

def prompt_collection(db: Database):
    while True:
        collection_list = db.list_collection_names()
        collectionName = iterfzf(collection_list, prompt="Select Collection To Modify > ")

        print(f'--- Selected {collectionName}')

        return collectionName

def modify_collection(db: Database, collectionName: str):
    metadata = get_dataset_metadata(db, collectionName)

    if(metadata == None):
        print(f'Unable to find metadata entry for collection "{collectionName}"...')
        print("Double check the collection name, and if correct please add it's metadata record")
        input()
        return

    print(f'Found metadata record')

    mappings: OrderedDict[str, AttributeMapping] = OrderedDict()

    # Create inital default mappings. These will be overriden with proper values
    # if we find a record later

    globalAttributes = get_global_attributes(db)

    for attribute in globalAttributes:
        mappings[attribute['name']] = AttributeMapping()

    if("globalAttributes" not in metadata):
        print("No previous existing global attribute data")
    else:
        print("Found existing global attributes mapping... loading")

        for a in metadata['globalAttributes']:
            isOfficalMapping = (a in mappings) and mappings[a].isOffical

            mappings[a] = AttributeMapping(oldMapping=metadata['globalAttributes'][a], isOfficalMapping=isOfficalMapping)

    print("Loading attribute list (This may take a while depending on the dataset size)")
    collectionKeys = get_collection_keys(db, collectionName)

    print()
    print("The current mappings are...")
    printMappings(mappings)



    while True:

        mappingIndexList = list(mappings.items())

        command = input("Enter attribute index or.. C(reate), D(elete), L(ist), H(elp), S(pecial), W(rite), Q(uit) > ")

        if command.isdigit():
            index = int(command)

            if(index >= len(mappingIndexList) or index < 0):
                print("Invalid Index")
                continue

            attribute = mappings[mappingIndexList[index][0]]

            modify_attribute(db, collectionName, mappingIndexList[index][0], attribute, collectionKeys)
        elif command.upper() == "C":
            toCreate = input("New Attribute Name > ")

            if toCreate in mappings:
                print(f'The attribute mapping "{toCreate}" already exists for collection "{collectionName}"')
            elif toCreate == "":
                print("Empty input. Not Creating")
            else:
                mappings[toCreate] = AttributeMapping(isOfficalMapping=False)
                print(f'Created attribute mapping "{toCreate}" on collection "{collectionName}"\n')
        elif command.upper() == "D":
            while True:
                toDelete = input("Index to delete > ")

                if toDelete == "":
                    break
                elif not toDelete.isdigit():
                    print("Invalid Input")
                    continue
                else:
                    index = int(toDelete)
                    key = mappingIndexList[index][0]

                    if confirm_input(f'Remove {key} mapping? {"(Its an official global attribute) " if mappingIndexList[index][1].isOffical else ""}', defNo=True):
                        if not mappingIndexList[index][1].isOffical or confirm_input("Are you sure? This is an offical mapping!!", defNo=True):
                            del mappings[mappingIndexList[index][0]]
                            print(f'Deleted attribute mapping for {key}')

                    print()
                    break
        elif command.upper() == "L":
            print()
            printMappings(mappings)
        elif command.upper() == "W":

            hasUnmapped: bool = False
            for mapping in mappingIndexList:
                if mapping[1].get_mapped_name() == None:
                    hasUnmapped = True
                    break

            if hasUnmapped:
                if not confirm_input(f'There are still unmapped entries. Are you sure you want to write your changes to the database?', defNo=True):
                    continue
            else:
                if not confirm_input(f'Are you sure you want to write your changes to the database?', defNo=True):
                    continue

            # Build the mapping object

            obj = {}

            for mapping in mappings.items():
               if mapping[1].mapping != None:
                obj[mapping[0]] = mapping[1].mapping

            print(obj)

            meta_collection: Collection = db["Metadata"]

            meta_collection.update_one({"collection": collectionName}, {'$set': {"globalAttributes": obj}})

            print("Wrote updated mappings to metadata record")

        elif command.upper() == "S":
            special_operation_dialog(db, collectionName, mappings)
        elif command.upper() == "Q":
            if confirm_input("You may have unsaved changes, are you sure you want to quit?", defNo=True):
                break
        elif command.upper() == "H":
            helpMessage = """
Help
----
C - Creates a new attribute
D - Deletes the attribute at a specific index
H - Displays this help message
L - List the current mappings
S - Perform a special operation
Q - Quit
W - Write metadata changes to database
"""
            print(helpMessage)

        else:
            print(f'Unknown command "{command}". View help (H) or enter valid command\n')

def modify_attribute(db: Database, collectionName: str, attributeName: str, attribute: AttributeMapping, possibleKeys: Dict[str, int]):

    while True:
        listOfLocalAttributes = [key for key, value in possibleKeys.items()]

        selectedAttribute = iterfzf(listOfLocalAttributes, prompt=f'Select a local collection attribute to map "{attributeName}" too > ')

        if selectedAttribute == None:
            return

        print()
        print_attribute_overview(db, collectionName, selectedAttribute)

        if not confirm_input(f'Are you sure you want to map the local attribute "{selectedAttribute}" to the global attribute "{attributeName}"?', defNo=True):
            if confirm_input(f'Would you like to continue trying to map the global attribute "{attributeName}"?', defNo=True):
                continue
            break

        print(f'Mapping global attribute {attributeName} to local attribute {selectedAttribute}')

        attribute.set_mapped_name(selectedAttribute)

        return

def print_attribute_overview(db: Database, collectionName: str, attributeName: str):
    collection = db[collectionName]

    attributeTypes = get_collection_attribute_types(collection, attributeName)

    query = {attributeName:{'$exists': True}}
    projection = {attributeName: 1}

    totalCount = collection.count_documents({})
    attributeCount = collection.count_documents(query)
    results = collection.find(query, projection).limit(10)

    print(f'The attribute "{attributeName}" is present in {attributeCount}/{totalCount} documents ({attributeCount/totalCount:.0%} of documents)')
    print(f'It has the following types...')

    for t in attributeTypes:
        print(f' - {t}')

    print()

    print("Some example values it has are...")

    for v in results:
        print(v[attributeName])

    print()

def special_operation_dialog(db: Database, collectionName: str, mappings: 'OrderedDict[str, AttributeMapping]'):
    print_special_operations()

    while True:
        mappingIndexList = list(mappings.items())

        cmd = input("Enter special operation (or H for help)> ")

        if cmd == "":
            break
        if cmd.upper() == "H":
            print_special_operations()
            continue

        if not cmd.isdigit():
            print("Invalid operation id")
            continue

        cmdi = int(cmd)

        if cmdi == 1: # Delete unoffical attributes
            # Create list of to-be-deleted attributes
            to_remove = [a for a in mappingIndexList if not a[1].isOffical]

            table = [[
                "" if mapping[1].get_mapped_name() != None else "?",
                "Yes" if mapping[1].isOffical else "No",
                mapping[0],
                mapping[1].get_mapped_name() or "None",
                str(mapping[1].mapping) # TODO: limit in future
                ] for mapping in to_remove]

            print(tabulate(table, headers=["Status", "Official", "Global Name", "Collection Name", "Actual Mapping Value"]))

            print("WARN: This operation will remove the above attributes from the mapping table\n")

            if confirm_input("Are you sure you want to *REMOVE* these attributes?", defNo=True):
                for mapping in to_remove:
                    del mappings[mapping[0]]
                print("Removed all unoffical mappings")
            break
        else:
            print(f'Unknown operation {cmdi}. Enter "H" to view list')
    pass

def print_special_operations():
    print("""Special Operations:
1. Delete unoffical attributes
""")

def confirm_input(prompt: str, defYes=False, defNo=False):
    while True:
        value = input(f'{prompt} ({"Y" if defYes else "y"}/{"N" if defNo else "n"}) > ')

        if(value == ""):
            if defYes:
                return True
            elif defNo:
                return False
            continue
        elif "YES".startswith(value.upper()):
            return True
        elif "NO".startswith(value.upper()):
            return False
        else:
            print("\nPlease select an option\n")

def printMappings(mappings: 'OrderedDict[str, AttributeMapping]'):
    table = [[
        "" if mapping[1].get_mapped_name() != None else "?",
        index,
        mapping[0],
        mapping[1].get_mapped_name() or "None",
        "Yes" if mapping[1].isOffical else "No",
        str(mapping[1].mapping) # TODO: limit in future
        ] for index, mapping in enumerate(mappings.items())]

    print(tabulate(table, headers=["Status", 'idx', "Global Name", "Collection Name", "Official", "Actual Mapping Value"]))

def main(verbose):
    try:
        db = mongo_client.get_database("scsc_data")

        collection = prompt_collection(db)

        if(collection == None):
            return

        modify_collection(db, collection)

    except ServerSelectionTimeoutError:
        print("Server not available.")
        sys.exit(1)
    except ConnectionFailure:
        print("Failed to connect to server.")
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        sys.exit(1)

def build_arg_parser():
    parser = argparse.ArgumentParser(
                prog='GlobalAttributeTool',
                description='Interactively generate global attribute mappings for dataset collections')

    # parser.add_argument('collectionName')
    # parser.add_argument('-w', '--write-out', action='store_true')    # writes out counts to database
    parser.add_argument('-v', '--verbose', action='store_true')      # print extra output
    return parser

if __name__ == "__main__":
    import sys

    parser = build_arg_parser()
    args = parser.parse_args()

    if len(sys.argv) < 1:
        print(f'Usage: python {sys.argv[0]}')
        sys.exit(1)
    else:
        main(args.verbose)