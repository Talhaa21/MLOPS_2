docker:
	docker build -t sample .

images:
	docker images

delete:
	docker rmi sample:latest 

containers:
	docker ps

start:
	docker run -d -p 8000:8000 --name my_app_container sample

stop:
	@docker stop $(CONTAINER_ID)
