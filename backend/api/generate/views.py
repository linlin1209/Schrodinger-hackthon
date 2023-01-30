from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import OpenApiExample
from drf_spectacular.utils import extend_schema
from api.generate.serializers import SmilesInputSerializer
from api.generate.examples import EXAMPLE_REQUEST
import science.science.GA


class smilesInput(APIView):
    serializer_class = SmilesInputSerializer

    def get(self, request):

        return Response(
            {
                "status": "OK",
            }
        )

    @extend_schema(
        examples=[
            OpenApiExample(name="Example", value=EXAMPLE_REQUEST, request_only=True)
        ]
    )
    def post(self, request):
        serializer = SmilesInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        return Response(
            {"smileExample": science.science.GA.run(serializer.validated_data)}
        )
