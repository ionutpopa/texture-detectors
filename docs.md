Task to resolve:
In clothes matching I encountered an issue of accidentally creating clothe matches for fabrics that don't go together and also that are too thick or textures would not fit properly.
To know details about the fabrics I created 3 models for the following tasks:
- fabric_class (denim, jersey, etc...)
- material_weight (lightweight, heavyweight, medium_weight)
- finish (shiny, matte, sheer, smooth, textures)

Resources:
Fabrics dataset obtain from: https://drive.google.com/file/d/1G_g3NEcluW9iKbWY6BiCMcSo0eLxCG0z/view
For annotations I used openAI CLIP

Run inference with this command:
python .\inference.py --model .\10_epochs_results\fabric_class_resnet50.pth --task fabric_class --image /path/to/image