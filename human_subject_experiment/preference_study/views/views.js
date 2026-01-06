var intro = {
    name: "intro",
    // introduction title
    title: "REDACTED",
    // introduction text
    text:
        "Thank you for participating in our study!<br><br>In this study, you will see 30 images, each paired with two potential descriptions of the image. Your task will be to determine which of the two descriptions you prefer. The whole study should take no longer than <strong>10 minutes</strong>.<br><br>Please do <strong>not</strong> participate on a mobile device since the page won't display properly.<br><small>If you have any questions or concerns, don't hesitate to contact me at REDACTED</small> ",
        // "Thank you for participating in our study!<br><br>In this study, you will see 30 images, each paired with two potential descriptions of the image. Your task will be to determine which of the two descriptions is more informative. The whole study should take no longer than <strong>12 minutes</strong>.<br><br>Please do <strong>not</strong> participate on a mobile device since the page won't display properly.<br><small>If you have any questions or concerns, don't hesitate to contact me at REDACTED</small> ",
    legal_info:
        "<strong>LEGAL INFORMATION</strong>:<br><br><strong>LEGAL INFORMATION</strong><br><br>Protocol Director:<br>REDACTED<br><br>Protocol Title:<br>REDACTED<br><br>IRB#<br>REDACTED<br><br>DESCRIPTION:<br>REDACTED<br><br>TIME INVOLVEMENT:<br>REDACTED<br><br>RISKS AND BENEFITS:<br>REDACTED<br><br>PAYMENTS:<br>REDACTED<br><br>PARTICIPANT’S RIGHTS:<br>REDACTED<br><br>CONTACT INFORMATION:<br>REDACTED<br><br>Please save or print a copy of this page for your records.<br><br>If you agree to participate in this research, please click “BEGIN EXPERIMENT”.",
    // introduction's slide proceeding button text
    buttonText: "Begin experiment",
    // render function renders the view
    render: function() {
        var viewTemplate = $("#intro-view").html();

        $("#main").html(
            Mustache.render(viewTemplate, {
                picture: "REDACTED",
                title: this.title,
                text: this.text,
                legal_info: this.legal_info,
                button: this.buttonText
            })
        );

        var prolificId = $("#prolific-id");
        var IDform = $("#prolific-id-form");
        var next = $("#next");

        var showNextBtn = function() {
            if (prolificId.val().trim() !== "") {
                next.removeClass("nodisplay");
            } else {
                next.addClass("nodisplay");
            }
        };

        if (config_deploy.deployMethod !== "Prolific") {
            IDform.addClass("nodisplay");
            next.removeClass("nodisplay");
        }

        prolificId.on("keyup", function() {
            showNextBtn();
        });

        prolificId.on("focus", function() {
            showNextBtn();
        });

        // moves to the next view
        next.on("click", function() {
            if (config_deploy.deployMethod === "Prolific") {
                exp.global_data.prolific_id = prolificId.val().trim();
            }

            exp.findNextView();
        });
    },
    // for how many trials should this view be repeated?
    trials: 1
};

var instructions = {
    name: "instructions",
    render: function(CT) {
        var viewTemplate = $("#instructions-view").html(); // don't think we even need mustache yet

        $("#main").html(
            Mustache.render(viewTemplate, {})
        );

        var next_button = $("#next");

        next_button.on('click', function () {
            exp.findNextView();
        });

    },
    trials: 1
};

var main = {
    name: "main",
    render: function(CT) {
        // fill variables in view-template
        console.log(exp.trial_info.main_trials[CT]);
        var viewTemplate = $("#main-view").html();

        let picture = exp.trial_info.main_trials[CT]['image_path'];
        let captions = exp.trial_info.main_trials[CT]['captions'];
        var caption_conditions = _.shuffle(Object.keys(captions));
        console.log(caption_conditions);
        console.log(captions[caption_conditions[0]]);
        // let question = "How likely is it for someone to say that <strong>" + person_term + "</strong> has <strong>" + attribute + "</strong>?";

        // for debugging:
        // let question = "How likely is it for someone to say that a <strong>woman</strong> has <strong>a beard</strong>?"

        $("#main").html(
            Mustache.render(viewTemplate, {
                picture: "trial_info/" + picture,
                descr1: captions[caption_conditions[0]],
                descr2: captions[caption_conditions[1]]
                // q1: exp.trial_info.main_trials[CT]['question'],
                // q1_slider_left: q1['sl_left'],
                // q1_slider_right: q1['sl_right']
            })
        );

        window.scrollTo(0,0);

        // var error = $('#error');
        // var next = $('#next');
        
        // functions
        function responses_complete() {
            return($('input[name=slider1]:checked').val() > 0)
        };

        // var q1_resp = $('input[name=slider1]:checked').val();

        var descr_selected = false;
        var selected_descr_pos = "None";
        var selections = [];

        // event functions
        $("#descr1_box").on("click", function(e) {
            console.log("descr 1 was clicked");
            descr_selected = true;
            selected_descr_pos = "left";
            selections.push(selected_descr_pos);
            $("#descr1_box").css({"border": "3px solid #66b032"});
            $("#descr2_box").css({"border": "2px solid #ccc"});
        })

        $("#descr2_box").on("click", function(e) {
            console.log("descr 2 was clicked");
            descr_selected = true;
            selected_descr_pos = "right";
            selections.push(selected_descr_pos);
            $("#descr1_box").css({"border": "2px solid #ccc"});
            $("#descr2_box").css({"border": "3px solid #66b032"});
        })

        $("#next").on("click", function(e) {
            // when input is selected, response and additional info stored in exp.trial_info
            if (!descr_selected) {
                $('#error').css({"display": "block"});
                // state = STATES.RESPOND;
                // respond_area.css({"display" : "inline"});
                // alt_text.css({"opacity": "1"});
                // comment_area.css({"display" : "inline"});
                // show_img.css({"display" : "block"});
                // instruction.text("Now answer the questions below!");
                // next.text("Continue!");
                // next.css({"display": "none"});
                // rt_article_read = Date.now();
            }
            else {
                rt_trial_done = Date.now();
                var trial_data = {
                    trial_number: CT + 1,
                    picture: picture,
                    selected_descr_position: selected_descr_pos,
                    selected_descr_condition: selected_descr_pos == "left" ? caption_conditions[0] : caption_conditions[1],
                    nonselected_descr_condition: selected_descr_pos == "left" ? caption_conditions[1] : caption_conditions[0],
                    selections: selections,
                    descr_left: captions[caption_conditions[0]],
                    descr_right: captions[caption_conditions[1]],
                    descr_left_condition: caption_conditions[0],
                    descr_right_condition: caption_conditions[1],
                    rt_trial: (rt_trial_done - startingTime) /1000
                };
                // console.log("FIRST TIME LOGGING THINGS!");
                // console.log((rt_trial_done - startingTime) /1000);
                // console.log(trial_data);

                exp.trial_data.push(trial_data);
                exp.findNextView();
            }
        })

        // record trial starting time
        var startingTime = Date.now();
    },
    trials: 30
};

var postTest = {
    name: "postTest",
    title: "Additional Info",
    text:
        "Answering the following questions is optional, but will help us understand your answers.",
    buttonText: "Continue",
    render: function() {
        var viewTemplate = $("#post-test-view").html();
        $("#main").html(
            Mustache.render(viewTemplate, {
                title: this.title,
                text: this.text,
                buttonText: this.buttonText
            })
        );

        $("#next").on("click", function(e) {
            // prevents the form from submitting
            e.preventDefault();

            // records the post test info
            exp.global_data.HitCorrect = $("#HitCorrect").val();
            exp.global_data.age = $("#age").val();
            // exp.global_data.education = $("#education").val();
            exp.global_data.languages = $("#languages").val();
            exp.global_data.enjoyment = $("#enjoyment").val();
            exp.global_data.comments = $("#comments")
                .val()
                .trim();
            // exp.global_data.difficulties = $("#difficulties")
            //     .val()
            //     .trim();
            exp.global_data.endTime = Date.now();
            exp.global_data.timeSpent =
                (exp.global_data.endTime - exp.global_data.startTime) / 60000;

            // moves to the next view
            exp.findNextView();
        });
    },
    trials: 1
};

var thanks = {
    name: "thanks",
    message: "Thank you for taking part in this experiment!",
    render: function() {
        var viewTemplate = $("#thanks-view").html();

        // what is seen on the screen depends on the used deploy method
        //    normally, you do not need to modify this
        if (
            config_deploy.is_MTurk ||
            config_deploy.deployMethod === "directLink"
        ) {
            // updates the fields in the hidden form with info for the MTurk's server
            $("#main").html(
                Mustache.render(viewTemplate, {
                    thanksMessage: this.message
                })
            );
        } else if (config_deploy.deployMethod === "Prolific") {
            $("main").html(
                Mustache.render(viewTemplate, {
                    thanksMessage: this.message,
                    extraMessage:
                        "Please press the button below to confirm that you completed the experiment with Prolific. Your completion code is C6F01LDX.<br />" +
                        "<a href=" +
                        config_deploy.prolificURL +
                        ' class="prolific-url">Confirm</a>'
                })
            );
        } else if (config_deploy.deployMethod === "debug") {
            $("main").html(Mustache.render(viewTemplate, {}));
        } else {
            console.log("no such config_deploy.deployMethod");
        }

        exp.submit();
    },
    trials: 1
};
