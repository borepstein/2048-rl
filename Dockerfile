FROM centos

WORKDIR /app

RUN yum update -y
RUN yum install python3 -y
RUN yum install python3-pip -y
RUN yum groupinstall 'Development Tools' -y
RUN yum install python3-devel -y
RUN yum install git -y
RUN yum install libjpeg-devel -y
RUN yum install wget -y
RUN wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-rhel8-11-5-local-11.5.0_495.29.05-1.x86_64.rpm
RUN rpm -i /app/cuda-repo-rhel8-11-5-local-11.5.0_495.29.05-1.x86_64.rpm
RUN rm -f /app/cuda-repo-rhel8-11-5-local-11.5.0_495.29.05-1.x86_64.rpm
RUN yum install --enablerepo=extras epel-release -y
RUN yum remove ipa-common ipa-common-client ipa-client -y
RUN yum update -y
RUN yum install kernel-debug-devel dkms -y
RUN yum install cuda -y
COPY pylintrc requirements.txt set_pythonpath.sh ./
COPY py_2048_rl/ ./py_2048_rl/
COPY experiments/ ./experiments/
RUN pip3 install -r requirements.txt
CMD /bin/bash -c "source ./set_pythonpath.sh; python3 ./py_2048_rl/learning/learning.py /data/learning"
