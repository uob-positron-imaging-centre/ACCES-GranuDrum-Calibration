# Docker image containing all Binder configuration + PICI-LIGGGHTS + Python libraries
FROM anicusan/pici-liggghts:v3.8.1-focal


# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}/*
USER ${NB_USER}
