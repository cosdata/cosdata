# Contributing to Cosdata

We’re excited you’re here and want to contribute to Cosdata, the next-gen vector database designed for AI-scale performance.

Whether you’re fixing a typo, writing tests, improving documentation, or building new features — thank you. Every contribution matters and helps improve the future of search and AI infrastructure.

---

# Contributor License Agreement (CLA)

Before contributing, **you must first read and agree to the [CLA](./CLA.md)**. No pull request will be accepted unless the contributor has agreed to the CLA.

---

# Getting Started

## 1. Initial Build Configuration

- ### Quick Install (Linux)
   To install Cosdata and all dependencies, run:
   
   ```bash
   curl -sL https://cosdata.io/install.sh | bash
   ```
   
   You should see something like this:
   
   <img width="626" alt="image" src="https://github.com/user-attachments/assets/b36b94a9-8eb0-488f-b870-5c045df314ea" />
   
   
   After installation, reload your shell configuration:
   
   ```bash
   source ~/.bashrc
   ```
   
   Finally, run the following command and provide your Admin Key when prompted:
   
   <img width="755" alt="image" src="https://github.com/user-attachments/assets/9bac0d6f-8b02-46ea-bac1-c472eb5e3fb9" />

- ### Install via Docker (macOS & Windows)

   **Step 1: Verify Docker is running**
   
   Check if Docker is properly installed and running by executing:
   
   ```bash
   docker --version
   ```
   
   You should see an output similar to the following:
   
   <img width="754" alt="image" src="https://github.com/user-attachments/assets/609182d7-6fa7-4c46-add9-46e71b6c2e3c" />
   
   **Step 2: Pull the latest Cosdata Docker image**
   
   Download the latest Cosdata image from Docker Hub:
   
   ```bash
   docker pull cosdatateam/cosdata:latest
   ```
   
   Expected output:
   
   <img width="670" alt="image" src="https://github.com/user-attachments/assets/83470227-0ce3-4418-9511-aeab8a4f6b5c" />
   
   **Step 3: Start the container by running the following command**
   
   <img width="754" alt="image" src="https://github.com/user-attachments/assets/1d04f981-a4bb-4566-8310-79898dc0943d" />


- ### Build from source
  Make sure you have Rust (v1.81.0+) & Cargo installed

   **Step 1: Fork the repository on GitHub**
   
   ![image](https://github.com/user-attachments/assets/3f1141b5-28a0-4bee-9fe4-d086ef4b71e0)
   
   **Step 2: Clone your fork to your machine**
   
  ```bash
  git clone https://github.com/YOUR_GITHUB_USERNAME/cosdata.git
  cd cosdata
  ```
   
   **Step 3: Build the project**
   
  ```bash
  cargo build
  ```
  
   **Step 4: Run a local instance**
  ```bash
  cargo build --release
  ./target/release/cosdata --admin-key YOUR_ADMIN_KEY
  ```

The server will be available at `http://localhost:8443`

> **Note**: If you face any problems, see the [Troubleshooting](#troubleshooting) section.

---

### 2. Branch Naming Convention

When creating a new branch, use the following naming convention:

Branch name: `GITHUB_HANDLE/ISSUE_NUMBER/SHORT_DESCRIPTION`

Example: `johndoe/42/fix-typo`

---

### 3. Join the Discord server

Before starting any work:

- Join our [Discord Server](https://discord.com/invite/qvm8FJJHPm) and briefly describe what you're working on.
- Ask if anyone else is already working on the issue.
- Get feedback or suggestions before implementation.

---

# Submitting Issues and Feature Requests:

- Please use the provided [Pull Request Template](https://github.com/cosdata/cosdata/blob/main/.github/PULL_REQUEST_TEMPLATE.md) when making a pull request. 
- Please use the provided [Issue Template](https://github.com/cosdata/cosdata/tree/main/.github/ISSUE_TEMPLATE) when reporting a bug or requesting a feature.
- Include screenshots, logs, or steps to reproduce where applicable.

---

# Pull Request Process

### Step-by-step Guide:

1. **Fork the repository** from [cosdata/cosdata](https://github.com/cosdata/cosdata).
2. **Clone your fork**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/cosdata.git
   cd cosdata
   ```

3. **Set up your branch**:

   ```bash
   git checkout -b YOUR_HANDLE/ISSUE_NUMBER/SHORT_DESCRIPTION
   ```

4. **Make changes and commit** with a meaningful message:

   Example: `fix: resolve bug in token parsing (#42)`

   Use:

   ```bash
   git add .
   git commit -m "fix: resolve bug in token parsing (#42)"
   ```

5. **Push your changes**:

   ```bash
   git push origin YOUR_BRANCH_NAME
   ```

6. **Create a pull request** on GitHub:
   - Fill out the PR template.
   - Link the related issue (e.g., `Closes #42`).
   - Wait for maintainers to review.

---

# Keeping your fork in sync

To keep your fork up-to-date with the main repository:

1. Add the main repo as a remote:

   ```bash
   git remote add upstream https://github.com/cosdata/cosdata.git
   ```

2. Fetch the latest changes:

   ```bash
   git fetch upstream
   ```

3. Rebase your branch on top of upstream/main:

   ```bash
   git checkout main
   git rebase upstream/main
   ```

4. Push changes to your fork:

   ```bash
   git push origin main
   ```

---

# Troubleshooting

- **Installation errors**: Make sure your environment meets all requirements (Rust, etc.).
- **Build issues**: Run `cargo clean && cargo build`.
- **Still stuck?** Reach out on our [Discord](https://discord.com/invite/qvm8FJJHPm).

---

Thank you for contributing!
