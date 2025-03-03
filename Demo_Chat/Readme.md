# Demo Chatbot


## Run API

To run the API replace your user id and put your hugggingface token in the `Dockerfile` and build the docker

```
docker build -t aixpademo .

```

When running the docker expose the port used by the API (default value 8013)
```
docker run -p 8013:8013 aixpademo
```

## Frontend

The frontend is a single html page `index_local.html` plus a folder with some documents that can be loaded in the page.

**Before loading  `index_local.html`** change the addess of the API at lines **`302`** and **`365`**


>[!IMPORTANT] 
>The API load the models only after the first request from the chatbot. To fully load the API load a document in the interface and send a message in che chat

