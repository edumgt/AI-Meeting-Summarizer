## open ai key 사용을 어떻게, KMS 등 AWS 의 리소스 사용

## docker 로 구성 하는데, 회사내부 용도 뿐 아니라 SaaS 구성 외부 사용자에게 플랫폼으로 제공하고자 하는데, AI 리소스가 용량이 큰데 어떻게 SaaS 로 구성할지
```
github-actions-ecr-push-role 생성용 스크립트를 추가했습니다: create_github_actions_ecr_role.sh (line 1)

이 스크립트는 아래를 처리합니다.

GitHub OIDC Provider 존재 여부 확인
github-actions-ecr-push-role 생성 또는 갱신
api, web 두 ECR 리포지토리에 대한 push 권한 inline policy 적용
마지막에 AWS_ROLE_ARN 출력
실행 예시는 이렇습니다.

cd /home/AI-Meeting-Summarizer

GITHUB_ORG=YOUR_ORG \
GITHUB_REPO=YOUR_REPO \
AWS_REGION=ap-northeast-2 \
ECR_API_REPOSITORY=ai-meeting-summarizer-api \
ECR_WEB_REPOSITORY=ai-meeting-summarizer-web \
./scripts/aws/create_github_actions_ecr_role.sh
PR까지 허용하려면 이렇게 실행하시면 됩니다.

ALLOW_PULL_REQUEST=true \
GITHUB_ORG=YOUR_ORG \
GITHUB_REPO=YOUR_REPO \
AWS_REGION=ap-northeast-2 \
ECR_API_REPOSITORY=ai-meeting-summarizer-api \
ECR_WEB_REPOSITORY=ai-meeting-summarizer-web \
./scripts/aws/create_github_actions_ecr_role.sh
이건 아직 제가 실제 AWS에 실행하진 않았습니다. 현재 세션에 AWS 계정 자격증명이 보장되지 않아서, 실행은 사용자 환경에서 하시는 게 안전합니다. 실행 후 출력되는 AWS_ROLE_ARN 값을 GitHub Secret에 넣으면 됩니다.


1개 파일 변경됨
실행 취소

리뷰
scripts/aws/create_github_actions_ecr_role.sh
aws cli 로 하나씩 차례로


아래 순서대로 aws cli로 하나씩 하면 됩니다. 값만 먼저 바꿔서 실행하세요.

사전 변수:

AWS_ACCOUNT_ID=123456789012
AWS_REGION=ap-northeast-2
GITHUB_ORG=YOUR_ORG
GITHUB_REPO=YOUR_REPO
ROLE_NAME=github-actions-ecr-push-role
ECR_API_REPOSITORY=ai-meeting-summarizer-api
ECR_WEB_REPOSITORY=ai-meeting-summarizer-web
1. GitHub OIDC Provider 확인

aws iam get-open-id-connect-provider \
  --open-id-connect-provider-arn arn:aws:iam::$AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com
없으면 생성:

aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
2. Trust policy 파일 생성

cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::$AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:$GITHUB_ORG/$GITHUB_REPO:ref:refs/heads/main"
          ]
        }
      }
    }
  ]
}
EOF
PR도 허용하려면 sub 배열에 이것도 추가:

"repo:YOUR_ORG/YOUR_REPO:pull_request"
3. Role 생성

aws iam create-role \
  --role-name $ROLE_NAME \
  --assume-role-policy-document file://trust-policy.json \
  --description "GitHub Actions role for pushing api/web images to ECR"
이미 있으면 trust policy만 갱신:

aws iam update-assume-role-policy \
  --role-name $ROLE_NAME \
  --policy-document file://trust-policy.json
4. ECR policy 파일 생성

cat > ecr-push-policy.json <<EOF
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
        "arn:aws:ecr:$AWS_REGION:$AWS_ACCOUNT_ID:repository/$ECR_API_REPOSITORY",
        "arn:aws:ecr:$AWS_REGION:$AWS_ACCOUNT_ID:repository/$ECR_WEB_REPOSITORY"
      ]
    }
  ]
}
EOF
5. Role에 policy 붙이기

aws iam put-role-policy \
  --role-name $ROLE_NAME \
  --policy-name ${ROLE_NAME}-ecr-push \
  --policy-document file://ecr-push-policy.json
6. 생성 확인

aws iam get-role --role-name $ROLE_NAME
7. GitHub Secret에 넣을 ARN

echo arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME
이 출력값을 AWS_ROLE_ARN으로 넣으면 됩니다.

원하시면 다음 답변에서 ECR repository 생성도 CLI로 차례대로 적어드리겠습니다.


ORG 사용 안함


그럼 ORG 대신 GitHub 사용자 계정을 넣으면 됩니다. GitHub OIDC의 sub는 조직 전용이 아니라 OWNER/REPO 형식이라서, 개인 계정 저장소면 OWNER가 곧 GitHub 아이디입니다.

예를 들어 GitHub 저장소가:

https://github.com/myname/AI-Meeting-Summarizer
이면 값은 이렇게 잡으면 됩니다.

GITHUB_OWNER=myname
GITHUB_REPO=AI-Meeting-Summarizer
Trust policy도 이렇게 바꾸면 됩니다.

cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::$AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:$GITHUB_OWNER/$GITHUB_REPO:ref:refs/heads/main"
          ]
        }
      }
    }
  ]
}
EOF
즉, 이제는 GITHUB_ORG 대신 GITHUB_OWNER만 쓰면 됩니다.

원하시면 제가 create_github_actions_ecr_role.sh (line 1)도 GITHUB_ORG 대신 GITHUB_OWNER 기준으로 바로 수정해드리겠습니다.


{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::086015456585:oidc-provider/token.actions.githubusercontent.com"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": "repo:/AI-Meeting-Summarizer:ref:refs/heads/main"
                }
            }
        }
    ]
} 에서 repo 의 owner 는 edumgt



맞습니다. 지금 sub 값이 잘못되어 있습니다.

현재:

"repo:/AI-Meeting-Summarizer:ref:refs/heads/main"
수정:

"repo:edumgt/AI-Meeting-Summarizer:ref:refs/heads/main"
전체는 이렇게 쓰시면 됩니다.

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::086015456585:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:edumgt/AI-Meeting-Summarizer:ref:refs/heads/main"
        }
      }
    }
  ]
}
이미 Role이 있다면 갱신은 이렇게 하시면 됩니다.

aws iam update-assume-role-policy \
  --role-name github-actions-ecr-push-role \
  --policy-document file://trust-policy.json
그리고 GitHub Actions에서 쓸 AWS_ROLE_ARN은 아마 이렇게 됩니다.

arn:aws:iam::086015456585:role/github-actions-ecr-push-role
원하시면 다음으로 ecr-push-policy.json도 계정 086015456585, owner/repo 기준과 맞는지 같이 점검해드릴게요.
```


## local, dev 환경에서 사용하는 env 
## prod 환경의 env , 서버주소를 반영


## API Hub 를 이용해서 회의록 자동화 모듈 가능한지 - https://aihub.or.kr/