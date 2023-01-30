from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import Descriptors
from selfies import decoder 
from selfies import encoder
import numpy as np
from copy import deepcopy
import itertools
from rdkit.Chem import QED
import random




def _combine_core_and_rgroups(core, rgroups):
    ATOM_PROP_ATOM_LABEL = 'atomLabel'
    """
    Helper function for rgroup enumeration
    """
    product_mols = []
    rgroup_names = rgroups.keys()
    product = Chem.RWMol(core)
    remove_count = 0
    for at in core.GetAtoms():
        if at.HasProp(ATOM_PROP_ATOM_LABEL):
            label = at.GetProp(ATOM_PROP_ATOM_LABEL)
            if label not in rgroup_names:
                continue

            # find where the rgroup will attach to the existing
            # product
            attach_idx = None
            remove_idx = None
            rgroup = Chem.RWMol(rgroups[label])
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        # attach_idx will go down by 1
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            if attach_idx is None:
                raise ValueError("Invalid rgroup provided")

            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)

    # clean labeled atoms
    product_clean = Chem.RWMol(product)
    remove_count = 0
    for at in product.GetAtoms():
        if at.HasProp(ATOM_PROP_ATOM_LABEL):
            product_clean.RemoveAtom(at.GetIdx()-remove_count)
            remove_count += 1
    return product_clean

def rgroup_enumerate(core_smi,rgroups):
    #print(rgroups_smiles)
    keys, values = zip(*rgroups.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    all_products = []
    for i in permutations_dicts:
        rgroups = {key:Chem.MolFromSmiles(i[key]) for key in i}
        product_mol = _combine_core_and_rgroups(Chem.MolFromSmiles(core_smi),rgroups)
        all_products.append(product_mol)
    return all_products

def mutations_random_grin(selfie, max_molecules_len, write_fail_cases=False):
    '''Return a mutated selfie string
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 50% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)
    
    while not valid:
        fail_counter += 1
                
        alphabet = ['[Branch1_1]', '[Branch1_2]','[Branch1_3]', '[epsilon]', '[Ring1]', '[Ring2]', '[Branch2_1]', '[Branch2_2]', '[Branch2_3]', '[F]', '[O]', '[=O]', '[N]', '[=N]', '[#N]', '[C]', '[=C]', '[#C]', '[S]', '[=S]', '[C][=C][C][=C][C][=C][Ring1][Branch1_1]']

        # Insert a character in a Random Location
        if np.random.random() < 0.5: 
            random_index = np.random.randint(len(chars_selfie)+1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index:]

        # Replace a random character 
        else:                         
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[random_index+1:]
            else:
                selfie_mutated_chars = chars_selfie[:random_index] + [random_character] + chars_selfie[random_index+1:]

                
        selfie_mutated = "".join(x for x in selfie_mutated_chars)
        sf = "".join(x for x in chars_selfie)
        try:
            smiles = decoder(selfie_mutated)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(smiles_canon) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    return (selfie_mutated, smiles_canon)


def crossover_random(selfie_best,selfie_m, write_fail_cases=False):
    '''Return a mutated selfie string
    
    Mutations are done until a valid molecule is obtained 
    Rules of mutation: With a 50% propbabily, either: 
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
    
    Parameters:
    selfie            (string)  : SELFIE string to be mutated 
    max_molecules_len (int)     : Mutations of SELFIE string are allowed up to this length
    write_fail_cases  (bool)    : If true, failed mutations are recorded in "selfie_failure_cases.txt"
    
    Returns:
    selfie_mutated    (string)  : Mutated SELFIE string
    smiles_canon      (string)  : canonical smile of mutated SELFIE string
    '''
    valid=False
    fail_counter = 0
    chars_selfie_best = get_selfie_chars(selfie_best)
    chars_selfie_m = get_selfie_chars(selfie_m)
    
    while not valid:
        fail_counter += 1
                
        # select random index for first mol
        random_index_best = np.random.randint(len(chars_selfie_best)+1)
        random_index_m = np.random.randint(len(chars_selfie_m)+1)
        selfie_crossover_chars = chars_selfie_best[:random_index_best]+chars_selfie_m[random_index_m:]


                
        selfie_crossover = "".join(x for x in selfie_crossover_chars)
        sf = "".join(x for x in chars_selfie_best)
        try:
            smiles = decoder(selfie_crossover)
            mol, smiles_canon, done = sanitize_smiles(smiles)
            if len(smiles_canon) > max_molecules_len or smiles_canon=="":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid=False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write('Tried to mutate SELFIE: '+str(sf)+' To Obtain: '+str(selfie_mutated) + '\n')
                f.close()
    return (selfie_crossover, smiles_canon)

def get_selfie_chars(selfie):
    '''Obtain a list of all selfie characters in string selfie
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns:
    chars_selfie: list of selfie characters present in molecule selfie
    '''
    chars_selfie = [] # A list of all SELFIE sybols from string selfie
    while selfie != '':
        chars_selfie.append(selfie[selfie.find('['): selfie.find(']')+1])
        selfie = selfie[selfie.find(']')+1:]
    return chars_selfie

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

def get_QED_prop(single_mol):
    data = dict()
    prop_names = ["MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS"]
    column_names = [
        "Image",
        "MW",
        "ALOGP",
        "HBA",
        "HBD",
        "PSA",
        "ROTB",
        "AROM",
        "ALERTS",                                                                                                                                                                                     
        "QED",
    ]
    p = single_mol
    property_dict = {'smiles':Chem.MolToSmiles(p)}
    props = QED.properties(p)
    ALL_QED_PROPS = ["MW", "ALOGP", "HBA", "HBD", "PSA", "ROTB", "AROM", "ALERTS"] 
    qed_props = ALL_QED_PROPS
    for prop in qed_props:
         a = float(props.__getattribute__(prop))
         property_dict[prop] = a
    property_dict["WEIGHTED_QED"] = round(QED.qed(p), 3)
    return property_dict
    
def run(input_dict):
    all_gen_products = []
    for core_smi in input_dict['core_smiles']:
        all_products_mol = rgroup_enumerate(core_smi,input_dict['rgroup_smiles'])
        #Draw.MolsToGridImage(all_products_mol)
        all_products_smi = [Chem.MolToSmiles(i).replace('~','-') for i in all_products_mol]
        gens = {0:all_products_smi}
        for current_gen in range(input_dict['num_gen']):
            childs = []
            parents = random.choices(gens[current_gen],k=input_dict['num_parents'])
            for i in parents:
                num_atom = Chem.MolFromSmiles(i).GetNumAtoms()
                parent_smi = i
                for j in range(input_dict['num_mutations']):
                    selfie_mutated, smiles_mutated = mutations_random_grin(encoder(parent_smi),num_atom+10)
                    childs.append(smiles_mutated)
            gens[current_gen+1] = childs
        for key in gens:
            all_gen_products += gens[key]
    products = [Chem.MolFromSmiles(smi) for smi in all_gen_products]
    # populate properties in to [ {'smile1':...,'MW':....},{'smile2':...,'MW':...}]
    all_properties = []
    for mol in products:
        props = get_QED_prop(mol) 
        all_properties.append(props)
    return all_properties

input_dict = {'core_smiles':["[*]C1CC([*])CC([*])O1 |$R1;;;;R2;;;R3;$|"],\
            'rgroup_smiles': {
                "R1" : ["*N1CCCC1", "*O"],
                "R2" : ["*C", "*C1CCCNC1"],
                "R3" : ["CC(-*)=O", "*-C1CCNC1"]
            },\
            'num_gen':2,\
            'num_mutations':6,\
            'num_parents':2
}

"""
core_smi = ["[*]C1CC([*])CC([*])O1 |$R1;;;;R2;;;R3;$|"]
rgroup_smiles = {
    "R1" : ["*N1CCCC1", "*O"],
    "R2" : ["*C", "*C1CCCNC1"],
    "R3" : ["CC(-*)=O", "*-C1CCNC1"]
}
# number of generations
num_gen = 2
# number of mutations
num_mut = 6
# number of parents in each generation
num_pat = 2
"""


all_properties = run(input_dict)
#print(all_properties)
print(len(all_properties))
