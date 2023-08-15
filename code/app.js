const express = require("express");
const bodyParser = require("body-parser");
const ejs = require("ejs");
const mysql = require("mysql2");
const https = require("https");
const { log } = require("console");

const app = express();
app.use(express.static("public"));
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({extended: true}));

const con = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "harsh123",
    database: "rec_sys"
});

app.get("/", function(req, res){
    res.render("home");
});

app.get("/login", function(req, res){
    res.render("login");
});

app.get("/register", function(req, res){
    res.render("register");
});

app.post("/register", function(req, res){
    const email = req.body.email;
    const user_name = req.body.username;
    const password = req.body.password;
    var link;
    con.connect(function(err){
        if (err){
            throw err;
            return;
        } 
        var sql = "insert into users(email, user_name, passwd) values(?, ?, ?)"
        con.query(sql, [email, user_name, password], function(error, result){
            if (error){
                throw error;
                return;
            } 
            const query = "perfume";
            const url = "https://api.unsplash.com/photos/random/?client_id=0HZjaF3jD29yUT3qZ7KlImk_DAHesDXjvyUJncdRCrg&query=" + query;
            console.log(url);
            https.get(url, function(response){
                response.on("data", function(data){
                    const productData = JSON.parse(data);
                    link = productData.urls.raw;
                })
            })
            res.render("index", {user_name: user_name, img1: link});
        })
    })
    
})

app.post("/login", function(req, res){
    const username = req.body.username;
    const password = req.body.password;
    var link1;
    var link2;
    var link3;
    var link4;
    var link5;
    var link6;
    var link7;
    var link8;
    con.connect(function(err){
        if(err){
            throw err;
            return;
        }
        var sql = "select passwd from users where user_name = ?";
        con.query(sql, [username], function(error, result){
            if (error){
                throw error;
                return;
            } 
            var passwd = result[0].passwd; 
            if (passwd === password){
                const query = "furniture";
                for (let i = 0; i < 8; i++) {
                    const url = `https://api.unsplash.com/photos/random/?client_id=0HZjaF3jD29yUT3qZ7KlImk_DAHesDXjvyUJncdRCrg&query=${query}`;
                    var links = [];
                    https.get(url, (result2) => {
                        var data = "";
                        result2.on("data", function(chunk) {
                            data += chunk;
                        });
                        result2.on("end", () => {
                            const productData = JSON.parse(data);
                            links[i] = productData.urls.raw;

                            // Check if all links have been fetched
                            if (links.filter(Boolean).length === 8) {
                                res.render("index", {
                                    user_name: username,
                                    img1: links[0],
                                    img2: links[1],
                                    img3: links[2],
                                    img4: links[3],
                                    img5: links[4],
                                    img6: links[5],
                                    img7: links[6],
                                    img8: links[7]
                                });
                            }
                        });
                    });
                }
            }
            else {
                res.render("error");
            }
        })
    })
})

app.listen(3000, function(){
    console.log("server started on port 3000");
});