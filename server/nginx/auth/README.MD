Put your `.htpasswd` file here & run:
    
    htpasswd -c .htpasswd <username>

To add another to an existing file (-c overwrites existing) run:
    
    `htpasswd -n` <username>

Then copy-paste the command-line result into the existing `.htpasswd` file. 
[More here](https://httpd.apache.org/docs/2.4/programs/htpasswd.html).

FYI, you may need to install `apache2-utils`, if you don't have it already

    $ sudo apt update && apt install apache2-utils