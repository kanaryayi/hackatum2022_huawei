#### Hackatum2022 Huawei Challenge
# OHero
### Context

1. Android Native Mobile App
2. Django Server
3. Pytorch CNN Pipeline

## Usage
Start server to handle image post requests and run CNN model to detect which class the environment belong. After runnning below commands HTTP requests should be done to base root endpoint of the server e.g 127.0.0.1:8000. Requests should contain raw byte image with Base64 encoding.
    
    cd server
    python3 manage.py runserver
    
## Testing Object Classification pipeline
### Training and Validation
After running below provide parent directory for the corresponding train/val dataset. 

    cd server
    python3 -m ml.training


    // e.g : path/to/parent/val
    //   path/to/parent/train
    //   CLI input should /path/to/parent
-------------------------------------------

### Inference
After running below provide parent directory for the corresponding issues dataset. 
      
    cd server
    python3 -m ml.testing


    // e.g : path/to/parent/issues.csv
    //   path/to/parent/*.jpg
    //   CLI input should be /path/to/parent
-------------------------------------------
