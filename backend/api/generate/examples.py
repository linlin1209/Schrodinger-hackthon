EXAMPLE_REQUEST = {
    "core_smiles": ["[*]C1CC([*])CC([*])O1 |$R1;;;;R2;;;R3;$|"],
    "rgroup_smiles": {
        "R1": ["*N1CCCC1", "*O"],
        "R2": ["*C", "*C1CCCNC1"],
        "R3": ["CC(-*)=O", "*-C1CCNC1"],
    },
    "num_gen": 2,
    "num_mutations": 6,
    "num_parents": 2,
}
