async function askQuestion(){

    const question=document.getElementById("question").value;

    if(question===""){

        alert("Please enter a question.");

        return;

    }

    document.getElementById("loading").style.display="block";

    document.getElementById("response").innerHTML="";

    try{

        const response=await fetch("/chat",{

            method:"POST",

            headers:{

                "Content-Type":"application/json"

            },

            body:JSON.stringify({

                question:question

            })

        });

        const data=await response.json();

        document.getElementById("loading").style.display="none";

        if(data.answer){

            document.getElementById("response").innerHTML=data.answer;

        }

        else if(data.response){

            document.getElementById("response").innerHTML=data.response;

        }

        else if(data.error){

            document.getElementById("response").innerHTML=data.error;

        }

        else{

            document.getElementById("response").innerHTML=
            JSON.stringify(data,null,2);

        }

    }

    catch(err){

        document.getElementById("loading").style.display="none";

        document.getElementById("response").innerHTML=err;

    }

}