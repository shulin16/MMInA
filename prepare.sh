#!/bin/bash

# prepare the evaluation
# re-validate login information
# mkdir -p ./.auth

export OPENAI_API_KEY=""
export SHOPPING="http://localhost:7770"
export SHOPPING_ADMIN="http://localhost:7780/admin"
export REDDIT="http://localhost:9999"
export GITLAB="http://localhost:8023"
export MAP="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000/"
export WIKIPEDIA="https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing"
export HOMEPAGE="http://localhost:4399" # this is a placeholder

sudo docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7770" # no trailing slash
sudo docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7770/" WHERE path = "web/secure/base_url";'
sudo docker exec shopping /var/www/magento2/bin/magento cache:flush

sudo docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780" # no trailing slash
sudo docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
sudo docker exec shopping_admin /var/www/magento2/bin/magento cache:flush

python browser_env/auto_login.py

# sudo docker exec shopping_admin mysql -u admin -p admin1234 -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
# export WIKIPEDIA="http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"