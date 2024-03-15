module.exports = {
  apps: [{
    name: 'MH-Commercial',
    script: 'C:/Users/V102645/AppData/Roaming/npm/node_modules/serve/bin/serve.js',
    args: "build -s --listen 4000",
    instances: 4,
    autorestart: true,
    watch: true,
    max_memory_restart: '2G',
    env: {
      NODE_ENV: 'development'
    },
    env_production: {
      NODE_ENV: 'production'
    }
  }]
};