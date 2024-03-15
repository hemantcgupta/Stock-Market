import React from 'react';
import axios from 'axios';
import eventApi from '../../API/eventApi'
import api from '../../API/api'
import APIServices from '../../API/apiservices'
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import Container from '@material-ui/core/Container';
import { images } from '../../Constants/images';
import Validator from '../../Constants/validator';
import _ from 'lodash';
import Swal from 'sweetalert2';
import Modal from 'react-bootstrap-modal';
import './login.scss';
import cookieStorage from '../../Constants/cookie-storage';
import config from '../../Constants/config';
import Constants from '../../Constants/validator';

const apiServices = new APIServices();

let error = Constants.getParameterByName('error', window.location.href)
error = error ? error : ''

class Login extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      email: '',
      password: '',
      emailMessage: '',
      passwordMessage: '',
      token: '',
      disable: true,
      loading: false,
      displayModal: false
    }
    this.textInput = React.createRef();
  }

  componentDidMount() {
    cookieStorage.deleteCookie();
    // setTimeout(() => {
    //   this.textInput.current.focus();
    // }, 800);
    this.registerloginButtonClick();
  }

  registerloginButtonClick = () => {
    document.addEventListener("keydown", (event) => {
      if (event.keyCode === 13) {
        this.login();
      }
    });
  }

  handleChange = e => {
    this.setState({ [e.target.name]: e.target.value })
    this._validate(e);
  }

  _validate = (e) => {
    const name = e.target.name;
    const value = e.target.value;
    let msg = ``;

    if (name === 'email') {
      let isValid = Validator.validateEmail(value);
      msg = value === '' ? `Username should not be empty*` : (!isValid ? `Please enter valid username` : ``);
      this.setState({ emailMessage: msg });
    }
    else if (name === 'password') {
      // let isValid = Validator.validatePassword(value);
      msg = value === '' ? `Password should not be empty*` : (value.length < 7 ? `Please enter valid password` : ``);
      this.setState({ passwordMessage: msg });
    }
    
    setTimeout(() => {
      this.formValid();
    }, 600);
  }

  formValid = () => {
    const { emailMessage, passwordMessage, email, password } = this.state;
    const errorCheck = _.isEmpty(emailMessage && passwordMessage)
    const emptyDataCheck = !_.isEmpty(email && password)
    if (errorCheck && emptyDataCheck) {
      this.setState({ disable: false })
    } else {
      this.setState({ disable: true })
    }
  }

  resetState = () => {
    this.setState({
      email: '',
      password: '',
      emailMessage: '',
      passwordMessage: '',
      disable: true,
      loading: false
    })
  }

  login = () => {
    error = ''
    this.setState({ loading: true })
    window.location.href = `${config.API_URL}/initiatelogin`
    // if (!this.state.disable) {
    //   this.setState({ loading: true })
    //   let data = {
    //     email: this.state.email,
    //     password: this.state.password
    //   }
    //   api.post('api/login', data)
    //     .then((response) => {
    //       this.redirectionToDashboard(response.data)
    //     })
    //     .catch((error) => {
    //       console.log('error', error.response)
    //       this.setState({ loading: false })
    //       if (error.response) {
    //         Swal.fire({
    //           title: 'Error!',
    //           text: error.response.data.error,
    //           icon: 'error',
    //           confirmButtonText: 'Ok'
    //         }).then(() => {
    //           this.resetState()
    //         })
    //       } else {
    //         Swal.fire({
    //           title: 'Error!',
    //           text: 'Something went wrong. Please try after some time',
    //           icon: 'error',
    //           confirmButtonText: 'Ok'
    //         }).then(() => {
    //           this.resetState()
    //         })
    //       }
    //     });
    // }
  }

  sendEvent = (token) => {
    var eventData = {
      event_id: "4",
      description: "User logged in into the system",
      where_path: "/",
      page_name: "Login Page"
    }
    eventApi.sendEventWithHeader(eventData, token)
  }

  redirectionToDashboard = (logInResponse) => {
    const token = `Token ${logInResponse.token}`
    eventApi.getMenusForDashboard(token)
      .then((response) => {
        this.setState({ loading: false });
        console.log('rahul');
        if (response) {
          if (response.data) {
            this.sendEvent(token);
            cookieStorage.createCookie('Authorization', `Token ${logInResponse.token}`, 1)
            cookieStorage.createCookie('userDetails', JSON.stringify(logInResponse), 1);
            cookieStorage.createCookie('menuData', JSON.stringify(response.data.menus), 1);
            cookieStorage.createCookie('events', JSON.stringify(response.data.events), 1);
            let menuData = response.data.menus;
            let dashboard = menuData.filter((data, i) => data.Name === "Dashboard")
            let path = dashboard[0].submenu[0].Path;
            this.props.history.push(`${path}`);
          }
        }
      })
      .catch((err) => {
        console.log('Menu Error', err);
      })
  }

  closeModal() {
    this.setState({ displayModal: false })
  }
  // const classes = useStyles();

  render() {
    return (
      <Container className={`login`} component="main" maxWidth="xs">
        <CssBaseline />
        <div className={`paper`}>
          <div className={`top-login`}>
            <Avatar className={`avatar`}>
              <LockOutlinedIcon />
            </Avatar>
            <Typography component="h1" variant="h5">
              Sign in
          </Typography>
          </div>
          {/* <TextField
            variant="outlined"
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            value={this.state.email}
            inputRef={this.textInput}
            onChange={(e) => this.handleChange(e)}
          /><p>{this.state.emailMessage}</p>
          <TextField
            variant="outlined"
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            autoComplete="current-password"
            value={this.state.password}
            onChange={(e) => this.handleChange(e)}
          /><p>{this.state.passwordMessage}</p> */}
          {/* <FormControlLabel
            control={<Checkbox value="remember" color="primary" />}
            label="Remember me"
          /> */}
          {/* For development */}
          <div className='logo-img'><img alt='logo' src={images.airline_logo} /></div>
          {/* For MH */}
          {/* <div className='logo-img'><img alt='logo' src={images.airline_logo} /></div> */}
          <Button
            // disabled={true}
            type={`submit`}
            fullWidth
            variant="contained"
            color="primary"
            className={`submit`}
            onClick={() => this.login()}
          >
            {this.state.loading ? <img src={images.ellipsis_loader} alt='' /> : `Click here to Login`}
          </Button>
          <p>{`${decodeURIComponent(error)}`}</p>
        </div>
      </Container>
    );
  }
}

export default Login;