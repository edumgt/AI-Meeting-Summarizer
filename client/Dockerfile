FROM node:24-alpine

WORKDIR /app

COPY client/package*.json /app/
RUN npm install

COPY client /app

EXPOSE 3000
CMD ["node", "server.js"]
