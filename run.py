from module import *

pipeline1 = pipeline(objdet_path="objdet_weights/harcasscade.xml",
                     imgcls_path="imgcls_weights/mobilenetv2_weights.keras",
                     source_path=0)
objdetmodel = pipeline1.load_objectdetection_model()
imgclsmodel = pipeline1.load_imageclassification_model()
database = pipeline1.load_database()
pipeline1.run(model1=objdetmodel,model2=imgclsmodel,database=database)




