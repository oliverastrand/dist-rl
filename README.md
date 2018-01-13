# Space Gorilas

## AWS instructions
In order to access the GPU instance, you'll need the `aws.pem` key pair (which is provided in the repository), and then:

    ssh -i "aws.pem" ec2-user@52.212.213.61

In order to activate the TensorFlow environment:

    source activate tensorflow_p36
    
The repository is cloned in the `/home` folder. If you need to update it again, just re-run `git pull` and provide your own credentials, they're not stored in any way (we can create a dummy github account if you don't feel confortable).
    
We can have concurrent users in the same instance, but we should still sync up because it can mess up the machine if intensive work is done concurrently in the machine.

### Important notice
To avoid unnecessary billing, the AWS instance should be stopped when it's not actively used. Ignore the warnings, it can be restarted without losing any data...
