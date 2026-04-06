#!/usr/bin/env bash

set -euo pipefail

ROLE_NAME="${ROLE_NAME:-github-actions-ecr-push-role}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
GITHUB_ORG="${GITHUB_ORG:-}"
GITHUB_REPO="${GITHUB_REPO:-}"
ECR_API_REPOSITORY="${ECR_API_REPOSITORY:-ai-meeting-summarizer-api}"
ECR_WEB_REPOSITORY="${ECR_WEB_REPOSITORY:-ai-meeting-summarizer-web}"
ALLOW_PULL_REQUEST="${ALLOW_PULL_REQUEST:-false}"

if [[ -z "${GITHUB_ORG}" || -z "${GITHUB_REPO}" ]]; then
  echo "GITHUB_ORG and GITHUB_REPO are required."
  echo "Example:"
  echo "  GITHUB_ORG=my-org GITHUB_REPO=AI-Meeting-Summarizer bash scripts/aws/create_github_actions_ecr_role.sh"
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required."
  exit 1
fi

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
OIDC_PROVIDER_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

if ! aws iam get-open-id-connect-provider \
  --open-id-connect-provider-arn "${OIDC_PROVIDER_ARN}" >/dev/null 2>&1; then
  echo "GitHub OIDC provider is missing in this AWS account."
  echo "Create it first in IAM:"
  echo "  Provider URL: https://token.actions.githubusercontent.com"
  echo "  Audience: sts.amazonaws.com"
  exit 1
fi

SUB_MAIN="repo:${GITHUB_ORG}/${GITHUB_REPO}:ref:refs/heads/main"

if [[ "${ALLOW_PULL_REQUEST}" == "true" ]]; then
  SUBJECTS_JSON=$(cat <<JSON
[
  "${SUB_MAIN}",
  "repo:${GITHUB_ORG}/${GITHUB_REPO}:pull_request"
]
JSON
)
else
  SUBJECTS_JSON=$(cat <<JSON
[
  "${SUB_MAIN}"
]
JSON
)
fi

TRUST_DOC="$(mktemp)"
POLICY_DOC="$(mktemp)"

cleanup() {
  rm -f "${TRUST_DOC}" "${POLICY_DOC}"
}
trap cleanup EXIT

cat > "${TRUST_DOC}" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "${OIDC_PROVIDER_ARN}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": ${SUBJECTS_JSON}
        }
      }
    }
  ]
}
JSON

cat > "${POLICY_DOC}" <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart"
      ],
      "Resource": [
        "arn:aws:ecr:${AWS_REGION}:${AWS_ACCOUNT_ID}:repository/${ECR_API_REPOSITORY}",
        "arn:aws:ecr:${AWS_REGION}:${AWS_ACCOUNT_ID}:repository/${ECR_WEB_REPOSITORY}"
      ]
    }
  ]
}
JSON

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "Role already exists. Updating assume-role policy."
  aws iam update-assume-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-document "file://${TRUST_DOC}"
else
  echo "Creating role ${ROLE_NAME}"
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "file://${TRUST_DOC}" \
    --description "GitHub Actions role for pushing api/web images to ECR"
fi

echo "Applying inline ECR policy to ${ROLE_NAME}"
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${ROLE_NAME}-ecr-push" \
  --policy-document "file://${POLICY_DOC}"

echo
echo "Done."
echo "AWS_ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
