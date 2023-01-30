import re
from rest_framework.serializers import Serializer
from rest_framework.serializers import ModelSerializer
from rest_framework.fields import CharField, ListField, IntegerField, DictField


class SmilesInputSerializer(Serializer):
    core_smiles = ListField(child=CharField(required=True), required=True)
    rgroup_smiles = DictField(
        child=ListField(child=CharField(required=True), required=True)
    )
    num_gen = IntegerField(required=True)
    num_mutations = IntegerField(required=True)
    num_parents = IntegerField(required=True)
