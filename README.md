How To Use Git and GitHub!!

My username on GitHub is wondest

  1. First, create a repository on GitHub: Here I create a repository named Simulator;

  2. Second, create a folder on your local computer and then start Gut Bash;

     GitBash: ssh-keygen -t rsa -C "396934200@qq.com"
     More: 396934200@qq.com is the registry email of GitHub
   
     It will generate two files, id_rsa and id_rsa.pub
   
  3. Open the SSH Setting of GitHub, add a ssh setting using the content of id_rsa.pub;
     More：Account settings”--“SSH Keys”
 
     Next test the connection between local and GitHub using the command in Gib Bash:
    
	 GitBash: git config --global user.name "Kinsh"
     GibBash: git config --global user.email "396934200@qq.com"
     GibBash: git config --global --list
     GitBash: ssh -T git@github.com
    
	 Prompt: You’ve successfully authenticated, but GitHub does not provide shell access.
    
    
   
  4. Pull the repository of GitHub
 
     GitBash: git clone https://git.github/wondest/Simulator.git
	
  5. Modify the files at local
  
  6. 
