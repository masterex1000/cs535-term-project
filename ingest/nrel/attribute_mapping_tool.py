import os
import argparse
from typing import Any, Dict, List
from collections import OrderedDict

from iterfzf import iterfzf
from tabulate import tabulate

from wrangle_utils import NRELDataSet

class AttributeMapping:
    def __init__(self, mapping=None, oldMapping = None, isExistingMapping = True):
        self.mapping: str = mapping
        self.oldMapping: str = oldMapping
        self.isExisting: bool = isExistingMapping

        if self.mapping == None:
            self.mapping = oldMapping

    def get_mapped_name(self) -> str:
        return self.mapping

    def set_mapped_name(self, newName: str):
        self.mapping = newName

def prompt_collection(dataset: NRELDataSet):
    while True:
        collection_list = dataset.get_sites() + ["quit"]
        collectionName = iterfzf(collection_list, prompt="Select Site Mappings To Modify > ")

        print(f'--- Selected {collectionName}')

        return collectionName
    
def modify_site(dataset: NRELDataSet, site_name: str, prior_mappings: Dict[str, Any]) -> Dict[str, Any]:
    mappings: OrderedDict[str, AttributeMapping] = OrderedDict()

    # Create initial default mappings. These will be overridden with proper values
    # if we find a record later

    globalAttributes = dataset.get_global_attributes()

    mappings = {}

    for attribute in globalAttributes:
        if attribute in prior_mappings:
            mappings[attribute] = AttributeMapping(oldMapping=prior_mappings[attribute], isExistingMapping=True)
        else:
            mappings[attribute] = AttributeMapping(isExistingMapping=False)

    collectionKeys = dataset.get_site_attributes(site_name)

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

            modify_attribute(dataset, site_name, mappingIndexList[index][0], attribute, collectionKeys)
        elif command.upper() == "C":
            toCreate = input("New Attribute Name > ")

            if toCreate in mappings:
                print(f'The attribute mapping "{toCreate}" already exists for site "{site_name}"')
            elif toCreate == "":
                print("Empty input. Not Creating")
            else:
                mappings[toCreate] = AttributeMapping(isExistingMapping=False)
                print(f'Created attribute mapping "{toCreate}" on site "{site_name}"\n')
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

                    if confirm_input(f'Remove {key} mapping? {"(Its an official global attribute) " if mappingIndexList[index][1].isExisting else ""}', defNo=True):
                        if not mappingIndexList[index][1].isExisting or confirm_input("Are you sure? This is an existing mapping!!", defNo=True):
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

            # obj = {}

            # for mapping in mappings.items():
            #    if mapping[1].mapping != None:
            #     obj[mapping[0]] = mapping[1].mapping

            # print(obj)

            # for mapping in mappings.items():
            #     if mapping[1].mapping != None:
            #         prior_mappings[]

            prior_mappings.clear()
            for mapping in mappings.items():
                if mapping[1].mapping != None:
                    prior_mappings[mapping[0]] = mapping[1].mapping

            # meta_collection: Collection = db["Metadata"]

            # meta_collection.update_one({"collection": collectionName}, {'$set': {"globalAttributes": obj}})

            # print("Wrote updated mappings to metadata record")

        elif command.upper() == "S":
            special_operation_dialog(mappings)
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

def modify_attribute(dataset: NRELDataSet, collectionName: str, attributeName: str, attribute: AttributeMapping, possibleKeys: List[str]):

    while True:
        # listOfLocalAttributes = [key for key, value in possibleKeys.items()]

        selectedAttribute = iterfzf(possibleKeys, prompt=f'Select a local collection attribute to map "{attributeName}" too > ')

        if selectedAttribute == None:
            return

        print()
        # print_attribute_overview(db, collectionName, selectedAttribute)

        if not confirm_input(f'Are you sure you want to map the local attribute "{selectedAttribute}" to the global attribute "{attributeName}"?', defNo=True):
            if confirm_input(f'Would you like to continue trying to map the global attribute "{attributeName}"?', defNo=True):
                continue
            break

        print(f'Mapping global attribute {attributeName} to local attribute {selectedAttribute}')

        attribute.set_mapped_name(selectedAttribute)

        return

def special_operation_dialog(mappings: 'OrderedDict[str, AttributeMapping]'):
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

        if cmdi == 1: # Delete unofficial attributes
            # Create list of to-be-deleted attributes
            to_remove = [a for a in mappingIndexList if not a[1].isExisting]

            table = [[
                "" if mapping[1].get_mapped_name() != None else "?",
                "Yes" if mapping[1].isExisting else "No",
                mapping[0],
                mapping[1].get_mapped_name() or "None",
                str(mapping[1].mapping) # TODO: limit in future
                ] for mapping in to_remove]

            print(tabulate(table, headers=["Status", "Official", "Global Name", "Collection Name", "Actual Mapping Value"]))

            print("WARN: This operation will remove the above attributes from the mapping table\n")

            if confirm_input("Are you sure you want to *REMOVE* these attributes?", defNo=True):
                for mapping in to_remove:
                    del mappings[mapping[0]]
                print("Removed all non-pre-existing mappings")
            break
        else:
            print(f'Unknown operation {cmdi}. Enter "H" to view list')
    pass

def print_special_operations():
    print("""Special Operations:
1. Delete non-pre-existing attributes
""")

def printMappings(mappings: 'OrderedDict[str, AttributeMapping]'):
    table = [[
        "" if mapping[1].get_mapped_name() != None else "?",
        index,
        mapping[0],
        mapping[1].get_mapped_name() or "None",
        "Yes" if mapping[1].isExisting else "No",
        str(mapping[1].mapping) # TODO: limit in future
        ] for index, mapping in enumerate(mappings.items())]

    print(tabulate(table, headers=["Status", 'idx', "Global Name", "Collection Name", "Is Existing", "Actual Mapping Value"]))

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

def main(verbose):
    dataset = NRELDataSet()

    mappings = dataset.load_mappings()

    while True:
        collection = prompt_collection(dataset)

        if(collection == None):
            return
        
        if collection == "quit":
            break

        
        if collection not in mappings:
            mappings[collection] = {}

        # existingMappings = mappings[collection] if collection in mappings else {}
        existingMappings = mappings[collection]


        modify_site(dataset, collection, existingMappings)

        dataset.save_mappings(mappings)

def build_arg_parser():
    parser = argparse.ArgumentParser(
                prog='GlobalAttributeTool',
                description='Interactively generate global attribute mappings for dataset collections')

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