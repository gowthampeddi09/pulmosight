FROM node:20-alpine

WORKDIR /app

# Copy package files first for layer caching
COPY package.json package-lock.json* ./

RUN npm install

COPY . .

EXPOSE 3000

# Use Next.js dev server (production would use `npm run build && npm start`)
CMD ["npm", "run", "dev"]
