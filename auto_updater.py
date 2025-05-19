import asyncio
import os
import subprocess
import sys

class AutoUpdater:
    """Auto-updater that pulls from metanova-labs/nova main branch and restarts."""
    
    UPDATE_INTERVAL = 30  # Check every hour
    REMOTE_URL = "https://github.com/metanova-labs/nova.git"
    BRANCH = "main"
    REPO_PATH = "."  # Current directory
    
    def __init__(self, logger):
        """Initialize with logger. Uses current directory as repo path."""
        self.logger = logger
        self._setup_remote()
    
    def _setup_remote(self):
        """Ensure remote URL is correct."""
        self.logger.info(f"Setting up remote URL: {self.REMOTE_URL}")
        
        returncode, stdout, stderr = self._run_git_command('remote', '-v')
        if 'origin' in stdout:
            returncode, stdout, stderr = self._run_git_command('remote', 'set-url', 'origin', self.REMOTE_URL)
        else:
            returncode, stdout, stderr = self._run_git_command('remote', 'add', 'origin', self.REMOTE_URL)
            
        if returncode != 0:
            self.logger.error(f"Failed to set up remote URL: {stderr}")
        
    def _run_git_command(self, *args):
        """Run git command and return results."""
        cmd = ['git'] + list(args)
        process = subprocess.run(
            cmd, 
            cwd=self.REPO_PATH,
            capture_output=True,
            text=True
        )
        return process.returncode, process.stdout.strip(), process.stderr.strip()
    
    def _reset_local_changes(self):
        """Reset local changes to HEAD."""
        self.logger.info("Resetting local changes before update")
        returncode, stdout, stderr = self._run_git_command('reset', '--hard', 'HEAD')
        if returncode != 0:
            self.logger.error(f"Failed to reset changes: {stderr}")
            return False
        return True
                
    def _check_for_updates(self):
        """Check if updates are available."""
        returncode, stdout, stderr = self._run_git_command('fetch', 'origin', self.BRANCH)
        if returncode != 0:
            self.logger.error(f"Failed to fetch updates: {stderr}")
            return False
                
        returncode, stdout, stderr = self._run_git_command('diff', f'HEAD..origin/{self.BRANCH}')
        if returncode != 0:
            self.logger.error(f"Failed to check if updates are available: {stderr}")
            return False
                
        return bool(stdout.strip()) 
                
    def _pull_updates(self):
        """Pull updates from remote branch."""
        self.logger.info(f"Pulling updates from origin/{self.BRANCH}")
        returncode, stdout, stderr = self._run_git_command('pull', 'origin', self.BRANCH)
        if returncode != 0:
            self.logger.error(f"Failed to pull updates: {stderr}")
            return False
        return True
                    
    def _restart_process(self):
        """Restart the process with same arguments."""
        self.logger.info(f"Restarting process with command: {' '.join(sys.argv)}")
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv[1:])
        except Exception as e:
            self.logger.error(f"Failed to restart process: {e}")
    
    async def start_update_loop(self):
        """Run update loop checking for and applying updates."""
        while True:
            try:
                self.logger.info(f"Checking for updates from {self.REMOTE_URL} ({self.BRANCH} branch)")
                
                if not self._reset_local_changes():
                    await asyncio.sleep(self.UPDATE_INTERVAL)
                    continue
                
                if self._check_for_updates():
                    self.logger.info("Updates available, pulling changes")
                    
                    if self._pull_updates():
                        self.logger.info("Updates successfully applied, restarting")
                        self._restart_process()
                        self.logger.error("Failed to restart after update")
                else:
                    self.logger.info("No updates available")
                    
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                
            self.logger.info(f"Next update check in {self.UPDATE_INTERVAL} seconds")
            await asyncio.sleep(self.UPDATE_INTERVAL) 